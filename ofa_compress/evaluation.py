import math
import torch
from collections import defaultdict
from utils import save_checkpoint
from criterions import AdjustLabelSmoothedCrossEntropyCriterion
import string
import time
import os
from data_utils.ofa_dataset import collate_tokens
from typing import Dict, List, Optional
from textbrewer.distiller_utils import move_to_device

TOKENIZER_PATH = "./tokenizer"

def get_perplexity(outputs=None, **kwargs):
    assert 'loss' in outputs
    perplexity = math.exp(torch.mean(outputs["loss"]))
    return {"perplexity": perplexity}


def ddp_evaluate(model, step, eval_dataloader, args, logger):
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info(f"Do predict in local rank : {args.local_rank}")
        evaluate(model, step, eval_dataloader, args, logger)
        args.evaluate_idx += 1
        if args.local_rank != -1:  # DDP is enabled
            torch.distributed.barrier()
    else:
        torch.distributed.barrier()


def evaluate(model, step, eval_dataloader, args, logger):
    try:
        model = model.module
    except:
        model = model
    tokenizer = args.tokenizer
    device = args.device

    if args.generator_version == "fairseq":
        try:
            generator = args.generator
        except:
            logger.info("Don't need generator when evaluation.")

    if args.rank == 0:
        logger.info("Evaluation...")
    model.eval()
    outputs = defaultdict(list)
    if args.task in ['caption_stage1', 'caption_stage2']:
        cider_sum = 0.0
        cider_cnt = 0
        forward_time = 0
        generate_time = 0
        for idx, data in enumerate(eval_dataloader):
            data = move_to_device(data, device)
            batch = data["net_input"]
            with torch.no_grad():
                forward_time_start = time.time()
                forward_time_end = time.time()
                forward_time += forward_time_end - forward_time_start
                forward_gen_start = time.time()
                if args.generator_version == "hf":
                    gen_output = model.generate(input_ids=batch["input_ids"],patch_images=batch["patch_images"],
                                                   patch_masks=batch["patch_masks"],
                                                   num_beams=args.beam,
                                                   max_length=args.max_len_b,
                                                   min_length=args.min_len,
                                                   no_repeat_ngram_size=args.no_repeat_ngram_size)
                elif args.generator_version == "fairseq":
                    with torch.no_grad():
                        gen_output = generator.generate([model], data)
                        gen_output = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

                forward_gen_end = time.time()
                generate_time += forward_gen_end-forward_gen_start
                decode_tokens = tokenizer.batch_decode(gen_output,skip_special_tokens=True)

                target_decode_tokens = tokenizer.batch_decode(data["target"],skip_special_tokens=True)
                hyps, refs = [], []
                transtab = str.maketrans({key: None for key in string.punctuation})
                for i in range(len(gen_output)):
                    hyps.append(decode_tokens[i].translate(transtab).strip())
                    refs.append(
                        [
                            sent.translate(transtab).strip()
                            for sent in target_decode_tokens[i].split('&&')
                        ]
                    )

                from metrics.cider import calculate_cider_scores
                scores = calculate_cider_scores(hyps, refs,args.CiderD_scorer)

                if idx % 100 == 0:
                    logger.info("example hypothesis: " + hyps[i])
                    logger.info("example reference: " + ' && '.join(refs[i]))
                cider_sum += scores.sum()
                cider_cnt += int(scores.size)
        eval_res = {"cider": cider_sum/cider_cnt}
        current_score = eval_res[args.metric]
        logger.info(
            f"Total generate time : {generate_time}; Second per sample : {generate_time/(cider_cnt+0.0000001)}; Total sample: {cider_cnt}"
        )
        if args.metric == "cider":
            if current_score > args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        else:
            if current_score < args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))

    elif args.task in ['refcoco','refcocog','refcocoplus']:
        score_sum = 0.0
        score_cnt = 0
        forward_time = 0
        generate_time = 0
        for idx, data in enumerate(eval_dataloader):
            data = move_to_device(data, device)
            batch = data["net_input"]
            with torch.no_grad():
                forward_time_start = time.time()
                forward_time_end = time.time()
                forward_time += forward_time_end - forward_time_start
                forward_gen_start = time.time()
                if args.generator_version == "hf":
                    constraint_start, constraint_end = args.constraint_range.split(',')
                    constraint_start = int(constraint_start)
                    constraint_end = int(constraint_end)
                    bad_words_ids = [[x] for x in list(range(4,constraint_start))+list(range(constraint_end,len(args.src_dict)))]
                    gen_output = model.generate(input_ids=batch["input_ids"], patch_images=batch["patch_images"],
                                                patch_masks=batch["patch_masks"],
                                                num_beams=args.beam,
                                                max_length=args.max_len_b+2,
                                                min_length=args.min_len+2,
                                                no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                bad_words_ids=bad_words_ids)
                    gen_output = [x[1:] for x in gen_output]
                elif args.generator_version == "fairseq":
                    with torch.no_grad():
                        gen_output = generator.generate([model], data)
                        gen_output = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

                forward_gen_end = time.time()
                generate_time += forward_gen_end-forward_gen_start

                hyps, refs = [], []
                for i in range(len(gen_output)):
                    hyps.append(gen_output[i][:-1] - len(args.src_dict) + args.num_bins)
                    refs.append(data["target"][i][:-1] - len(args.src_dict) + args.num_bins)

                if idx % 100 == 0:
                    logger.info(f"example hypothesis: {hyps[0]}"  )
                    logger.info(f"example reference: {refs[0]}" )

                hyps_tensor, refs_tensor = torch.stack(hyps, dim=0), torch.stack(refs, dim=0)
                hyps_tensor = hyps_tensor / (args.num_bins - 1) * args.max_image_size
                refs_tensor = refs_tensor / (args.num_bins - 1) * args.max_image_size
                hyps_tensor[:, ::2] /= data['w_resize_ratios'].unsqueeze(1)
                hyps_tensor[:, 1::2] /= data['h_resize_ratios'].unsqueeze(1)
                refs_tensor[:, ::2] /= data['w_resize_ratios'].unsqueeze(1)
                refs_tensor[:, 1::2] /= data['h_resize_ratios'].unsqueeze(1)


                def calculate_ap_score(hyps, refs, thresh=0.5):
                    interacts = torch.cat(
                        [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
                         torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
                        dim=1
                    )
                    area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
                    area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
                    interacts_w = interacts[:, 2] - interacts[:, 0]
                    interacts_h = interacts[:, 3] - interacts[:, 1]
                    area_interacts = interacts_w * interacts_h
                    ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
                    return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

                scores = calculate_ap_score(hyps_tensor, data['region_coords'].float())
                score_sum += scores.sum().item()
                score_cnt += scores.size(0)
        eval_res = {"ap": score_sum/score_cnt}
        current_score = eval_res[args.metric]
        logger.info(
            f"Total generate time : {generate_time}; Second per sample : {generate_time/(score_cnt)}"
        )
        logger.info(
            f"Total score : {score_sum}; Total sample: {score_cnt}"
        )
        if args.metric == "ap":
            if current_score > args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        else:
            if current_score < args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))

    elif args.task in ['snli_ve']:
        score_sum = 0.0
        score_cnt = 0
        for idx, data in enumerate(eval_dataloader):
            data = move_to_device(data, device)
            batch = data["net_input"]
            with torch.no_grad():
                encoder_out = model.encoder(
                    batch["input_ids"],
                    patch_images=batch["patch_images"],
                    patch_masks=batch["patch_masks"]
                )
                valid_result = []
                eos_item = torch.LongTensor([tokenizer.eos_token_id])
                for valid_answers, valid_constraint_masks in zip(args.valid_answers_list, args.valid_constraint_masks_list):
                    valid_size = len(valid_answers)
                    valid_tgt_items = [
                        torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
                        for decoder_prompt in data["decoder_prompts"] for valid_answer in valid_answers
                    ]
                    valid_prev_items = [
                        torch.cat([torch.tensor(decoder_prompt), valid_answer])
                        for decoder_prompt in data["decoder_prompts"] for valid_answer in valid_answers
                    ]
                    valid_constraint_mask_items = [
                        torch.cat(
                            [torch.zeros(len(decoder_prompt) - 1, valid_constraint_mask.size(1)).bool(),
                             valid_constraint_mask],
                            dim=0
                        )
                        for decoder_prompt in data["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
                    ]
                    valid_tgt = collate_tokens(valid_tgt_items, pad_idx=tokenizer.pad_token_id).to(device)
                    valid_prev_output = collate_tokens(valid_prev_items, pad_idx=tokenizer.pad_token_id).to(device)
                    valid_constraint_masks = collate_tokens(valid_constraint_mask_items, pad_idx=tokenizer.pad_token_id).to(device)

                    new_encoder_out = {}
                    new_encoder_out["last_hidden_state"] = encoder_out["last_hidden_state"].repeat_interleave(valid_size, dim=0)
                    new_encoder_out["padding_mask"] = encoder_out["padding_mask"].repeat_interleave(valid_size, dim=0)
                    new_encoder_out["position_embedding"] = encoder_out["position_embedding"].repeat_interleave(valid_size, dim=0)


                    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
                        r"""
                        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
                        """
                        bsz, src_len = mask.size()
                        tgt_len = tgt_len if tgt_len is not None else src_len

                        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
                        return expanded_mask.masked_fill(expanded_mask.bool(), torch.finfo(dtype).min)

                    encoder_attention_mask = _expand_mask(new_encoder_out["padding_mask"], new_encoder_out["last_hidden_state"].dtype,
                                                          valid_prev_output.shape[-1])

                    decoder_out = model.decoder(valid_prev_output, encoder_hidden_states=new_encoder_out["last_hidden_state"],
                                                encoder_attention_mask=encoder_attention_mask,
                                                src_pos_embed=new_encoder_out["position_embedding"]
                                                )

                    decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
                    lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
                    scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
                    scores = scores.masked_fill(valid_tgt.eq(tokenizer.pad_token_id), 0)
                    scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
                    scores = scores.sum(1)
                    scores = scores.view(-1, valid_size)
                    valid_result.append(scores)
            valid_result = torch.cat(valid_result, dim=-1)
            predicts = valid_result.argmax(1).tolist()
            hyps = [args.index2ans[predict_index] for predict_index in predicts]
            results = [{"uniq_id": id, "answer": hyp} for id, hyp in zip(data["id"].tolist(), hyps)]
            scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(data['ref_dict'], hyps)]
            score_sum += sum(scores)
            score_cnt += len(scores)
            if idx % 100 == 0:
                logger.info(f"example hypothesis: {hyps[0]}")
                logger.info(f"example reference: {data['ref_dict'][0]}")
        score = score_sum / score_cnt
        score = score if isinstance(score, float) else score.item()
        score = round(score, 4)
        eval_res = {"acc": score}
        current_score = eval_res[args.metric]
        logger.info(
            f"Total sample: {score_cnt}"
        )
        if args.metric == "acc":
            if current_score > args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        else:
            if current_score < args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))

    elif args.task in ['vqa_gen']:
        score_sum = 0.0
        score_cnt = 0
        for idx, data in enumerate(eval_dataloader):
            data = move_to_device(data, device)
            batch = data["net_input"]
            with torch.no_grad():
                if args.val_inference_type == "beamsearch":
                    hypos = args.generator.generate([model], data, prefix_tokens=data['prefix_tokens'])
                    hypos = [hypos[i][0]["tokens"] for i in range(len(hypos))]
                    results = []
                    for i, sample_id in enumerate(data["id"].tolist()):
                        prefix_len = data['prefix_tokens'][i].ne(1).sum().item()
                        detok_hypo_str = tokenizer.batch_decode(hypos[i][0]["tokens"][prefix_len:], skip_special_tokens=True)
                        results.append({"question_id": int(sample_id), "answer": detok_hypo_str.strip()})
                    scores = [ref_dict.get(result['answer'], 0) for ref_dict, result in
                              zip(data['ref_dict'], results)]
                elif args.val_inference_type == "allcand":
                    encoder_out = model.encoder(
                        batch["input_ids"],
                        patch_images=batch["patch_images"],
                        patch_masks=batch["patch_masks"]
                    )
                    eos_item = torch.tensor([tokenizer.eos_token_id])
                    pad = tokenizer.pad_token_id
                    valid_result = []
                    for valid_answers, valid_constraint_masks in zip(args.valid_answers_list,
                                                                     args.valid_constraint_masks_list):
                        valid_size = len(valid_answers)
                        valid_tgt_items = [
                            torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
                            for decoder_prompt in data["decoder_prompts"] for valid_answer in valid_answers
                        ]
                        valid_prev_items = [
                            torch.cat([torch.tensor(decoder_prompt), valid_answer])
                            for decoder_prompt in data["decoder_prompts"] for valid_answer in valid_answers
                        ]
                        valid_constraint_mask_items = [
                            torch.cat([torch.zeros(len(decoder_prompt) - 1, valid_constraint_mask.size(1)).bool(),
                                       valid_constraint_mask], dim=0)
                            for decoder_prompt in data["decoder_prompts"] for valid_constraint_mask in
                            valid_constraint_masks
                        ]
                        valid_tgt = collate_tokens(valid_tgt_items, pad_idx=pad, left_pad=False).to(device)
                        valid_prev_output = collate_tokens(valid_prev_items, pad_idx=pad, left_pad=False).to(device)
                        valid_constraint_masks = collate_tokens(valid_constraint_mask_items, pad_idx=pad,
                                                                left_pad=False).to(device)
                        new_encoder_out = {}
                        new_encoder_out["last_hidden_state"] = encoder_out["last_hidden_state"].repeat_interleave(valid_size, dim=0)

                        new_encoder_out["padding_mask"] = encoder_out["padding_mask"].repeat_interleave(valid_size, dim=0)

                        new_encoder_out["position_embedding"] = encoder_out["position_embedding"].repeat_interleave(valid_size, dim=0)


                        def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
                            r"""
                            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
                            """
                            bsz, src_len = mask.size()
                            tgt_len = tgt_len if tgt_len is not None else src_len

                            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
                            return expanded_mask.masked_fill(expanded_mask.bool(), torch.finfo(dtype).min)

                        encoder_attention_mask = _expand_mask(new_encoder_out["padding_mask"],
                                                              new_encoder_out["last_hidden_state"].dtype,
                                                              valid_prev_output.shape[-1])

                        decoder_out = model.decoder(valid_prev_output, encoder_hidden_states= new_encoder_out["last_hidden_state"],
                                                    encoder_attention_mask=encoder_attention_mask,
                                                    src_pos_embed=new_encoder_out["position_embedding"]
                                                    )
                        decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
                        lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
                        scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
                        scores = scores.masked_fill(valid_tgt.eq(tokenizer.pad_token_id), 0)
                        scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
                        scores = scores.sum(1)
                        scores = scores.view(-1, valid_size)
                        valid_result.append(scores)
                    valid_result = torch.cat(valid_result, dim=-1)
                    predicts = valid_result.argmax(1).tolist()
                    hyps = [args.index2ans[predict_index] for predict_index in predicts]
                    scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(data['ref_dict'], hyps)]
                score_sum += sum(scores)
                score_cnt += len(scores)
                if idx % 100 == 0:
                    logger.info(f"example hypothesis: {hyps[0]}")
                    logger.info(f"example reference: {data['ref_dict'][0]}")
        score = score_sum / score_cnt
        score = score if isinstance(score, float) else score.item()
        score = round(score, 4)
        eval_res = {"acc": score}
        current_score = eval_res[args.metric]
        logger.info(
            f"Total sample: {score_cnt}"
        )
        if args.metric == "acc":
            if current_score > args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        else:
            if current_score < args.best_score:
                args.best_score = current_score
                args.best_step = step
                save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step"))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))

    elif args.task in ['pretrain']:
        losses = []
        sample_size = 0
        for idx, data in enumerate(eval_dataloader):
            data[0] = move_to_device(data[0], device)
            data[1] = move_to_device(data[1], device)
            batch_v1 = data[0]["net_input"]
            batch_v2 = data[1]["net_input"]
            with torch.no_grad():
                outputs = [model(**batch_v1), model(**batch_v2)]
                criterion = AdjustLabelSmoothedCrossEntropyCriterion(args)
                loss, sample_size_i, logging_output = criterion(outputs, data)
                losses.append(loss.unsqueeze(0))
                sample_size += sample_size_i
        loss = torch.cat(losses, dim=0)
        loss = torch.sum(loss)
        eval_res = {"loss": loss / sample_size}
        current_score = eval_res[args.metric]
        if current_score < args.best_score:
            args.best_score = current_score
            args.best_step = step
            save_checkpoint(step, model, args=args, best=True)
        if torch.distributed.get_rank() == 0:
            idx = args.evaluate_idx % args.keep_last_ckpt_num
            model.save_pretrained(os.path.join(args.output_dir, "saved_mode_step_%d" % idx))
            print(
                'global rank {} is saving checkpoint at iteration {:7d}'.
                    format(torch.distributed.get_rank(), step))
    else:
        raise NotImplementedError(
            f"The eval metric of task {args.task} is not implemented.")

    res_string = "\t".join([f"{k}: {v:.6f}" for k, v in eval_res.items()])

    if args.rank == 0:
        logger.info(f"Eval results at {step} steps:")
        logger.info(f"{res_string}")
        logger.info(
            f"Current best {args.metric}: {args.best_score} at {args.best_step} step"
        )
    model.train()
