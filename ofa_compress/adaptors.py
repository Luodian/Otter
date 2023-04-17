from criterions import AdjustLabelSmoothedCrossEntropyCriterion

class FinetuneAdaptor(object):
    """
    The adaptor that compute the hard label loss.
    """
    def __init__(self, args):
        self.batch = None
        self.model_outputs = None
        self.args = args


    def __call__(self, batch, model_outputs):
        outputs = {}
        criterion = AdjustLabelSmoothedCrossEntropyCriterion(self.args)

        loss, sample_size, logging_output = criterion(model_outputs, batch)
        outputs["losses"] = loss/logging_output['sample_size']
        outputs["sample_size"] = logging_output['sample_size']
        outputs["target"] = batch["target"]
        if "constraint_masks" in batch:
            outputs["constraint_masks"] = batch["constraint_masks"]

        for k1, k2 in zip(["encoder_attentions", "decoder_attentions",
                           "encoder_hidden_states","decoder_hidden_states",
                           "encoder_last_hidden_state", "logits",
                           "cross_attentions"],
                          ["encoder_attention", "decoder_attention",
                           "encoder_hidden", "decoder_hidden",
                           "encoder_last", "logits",
                           "cross_attention"]):
            if k1 in model_outputs:
                outputs[k2] = model_outputs[k1]
        return outputs

class PretrainAdaptor(object):
    """
    The adaptor that compute the hard label loss.
    """
    def __init__(self, args):
        self.batch = None
        self.model_outputs = None
        self.args = args


    def __call__(self, batch, model_outputs):
        outputs = {}
        criterion = AdjustLabelSmoothedCrossEntropyCriterion(self.args)

        loss, sample_size, logging_output = criterion(model_outputs, batch)
        outputs["losses"] = loss
        outputs["sample_size"] = logging_output['sample_size']

        for k1, k2 in zip(["encoder_attentions", "decoder_attentions",
                           "encoder_hidden_states","decoder_hidden_states",
                           "encoder_last_hidden_state", "logits"],
                          ["encoder_attention", "decoder_attention",
                           "encoder_hidden", "decoder_hidden",
                           "encoder_last", "logits"]):
            for i in range(2):
                if k1 in model_outputs[i]:
                    outputs[k2+'_%d' % i] = model_outputs[i][k1]
        return outputs