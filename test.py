# from tokenizer.bpe import get_encoder

prompt = 'Initialize grid.'

# tokenizer = get_encoder()
# token = tokenizer.encode(prompt)
# print(token)

# token, mask = tokenizer.padded_tokens_and_mask(
#     token, 64
# )

# print(token)
# print(tokenizer.n_vocab)

from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english")
token = tokenizer(prompt)
print(token)