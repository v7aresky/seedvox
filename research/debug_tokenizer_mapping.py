from seedvox.utils.tokenizer import PhonemeTokenizer
tokenizer = PhonemeTokenizer()
# The IDs returned by PhoneticAligner in the logs were:
# [34, 62, 32, 34, 71, 27, 3, ...]
# Let's see what these map to.
for i in [34, 62, 32, 34, 71, 27, 3]:
    print(f"ID {i}: {tokenizer.id_to_ph.get(i, '?')}")
