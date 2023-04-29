import sys

import transformers
from transformers import LlamaTokenizer
from ofa_compress.arguments import add_data_args

from collie_core.train.params import parse_args
from collie_core.train.data import get_data
from flamingo_hf import FlamingoModel

def main(args):
    parser = parse_args(args)
    parser = add_data_args(parser)
    args = parser.parse_args()
    # model = FlamingoModel.from_pretrained(args.flamingo_path)
    # tokenizer = model.text_tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("luodian/llama-7b-hf")
    multi_instruct_dataset = get_data(args, image_processor=None, tokenizer=tokenizer, dataset_type="multi_instruct")

if __name__ == "__main__":
    main(sys.argv[1:])