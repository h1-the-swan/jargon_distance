Jargon Distance code (Python)

Jason Portenoy 2018

Initialize a JargonDistance instance with a `term_counts` dict and (optionally) a `group_map` dict.
`term_counts` is a mapping of document -> term counter (Counter object).

```
from jargon_distance import JargonDistance
j = JargonDistance(term_counts)
```

Then, calculate the jargon distances:

```
j.calculate_jargon_distance()
j.write_to_file('jargon_distance.csv')
```

See `demo.ipynb` for a more detailed example.

![Dendrogram showing the jargon distance between different works of literature.](literature_dendrogram.png){#fig:dendrogram}
