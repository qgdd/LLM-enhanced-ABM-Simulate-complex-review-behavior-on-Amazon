from rouge_metric import PerlRouge

# Initialize the ROUGE metric
rouge = PerlRouge(rouge_n_max=3, rouge_l=True, rouge_w=True,
    rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)

# The actual review (reference)
references = [[
    "These jeans fit very well and are tight in all the right places and give just enough to not rub anywhere or ever be uncomfortable. Need a pair of nice white jeans for an event? These are priced right and fit very well."
]]

# The generated summary (hypothesis)
hypotheses = [
    "Wear em In Comfort"
]

# Evaluate the summary using the ROUGE metric
scores = rouge.evaluate(hypotheses, references)
print(scores)
