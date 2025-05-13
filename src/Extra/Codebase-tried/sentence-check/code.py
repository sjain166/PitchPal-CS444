# NOT WORKING

from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1          # CPU for this demo; set to 0 if you have GPU
)

text = (
    "The history of renewable solar energy is a story of scientific curiosity evolving into one of the greatest technological pursuits of the 21st century, beginning with the 19th-century discovery of the photovoltaic effect by French physicist Edmond Becquerel, which revealed that certain materials could convert light directly into electricity, and progressing through Charles Fritts’s 1880s selenium solar cells, which, despite their mere one-percent efficiency, laid the conceptual groundwork for modern photovoltaics. By the mid-20th century, Bell Labs had produced the first practical silicon-based solar cell, achieving six-percent efficiency and powering a small toy Ferris wheel as a public demonstration, a milestone that coincided with the dawn of the space race, where satellites such as Vanguard 1 depended on solar arrays for sustained operation beyond Earth’s atmosphere. Over subsequent decades, governments, research institutions, and private companies sought to reduce manufacturing costs and increase efficiency, developing innovations like passivated-emitter rear-contact (PERC) cells, thin-film cadmium-telluride modules, and perovskite tandem architectures, each iteration edging photovoltaics closer to parity with fossil-fuel-generated electricity. Meanwhile, large-scale deployment accelerated: Germany’s Energiewende policy catalyzed gigawatts of residential rooftop installations; China’s Belt and Road Initiative funded vast desert solar farms like the Tengger “Great Wall” array; and utility-scale projects in the American Southwest, such as the 579-MW Solar Star facility, demonstrated the feasibility of integrating intermittent renewable power into aging grids through advances in power electronics, smart-inverter technology, and lithium-ion storage. Today, the solar industry faces the dual imperative of scaling sustainably and mitigating supply-chain constraints—from polysilicon purity bottlenecks to ethical sourcing of silver and rare-earth elements—while researchers explore next-generation concepts like luminescent solar concentrators embedded in building windows, photovoltaic-thermal hybrids that co-generate heat and electricity, and bioinspired light-harvesting complexes modeled after photosynthetic bacteria. Despite lingering challenges in recycling end-of-life panels and ensuring grid resiliency against variable output, the cumulative global PV capacity surpassed 1 terawatt in 2023, propelling a virtuous cycle of cost decline and adoption that positions solar energy not merely as an alternative but as a cornerstone of the decarbonized energy systems envisioned in the Paris Agreement and beyond."
)  # ~40 tokens → fine for a summary

summary = summarizer(
    text,
    max_length=40,     # upper bound of tokens in the summary
    min_length=10,     # lower bound
    do_sample=False,   # deterministic
    clean_up_tokenization_spaces=True
)[0]['summary_text']

print(summary)

# The history of renewable solar energy is a story of scientific curiosity evolving into one of the greatest technological pursuits of the 21st century. By the mid-20th century, Bell Labs had