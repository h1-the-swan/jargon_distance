Jargon Distance code (Python)

Jason Portenoy 2018

Initialize a JargonDistance instance with a term_counts dict and (optionally) a group_map dict:

```
from jargon_distance import JargonDistance
j = JargonDistance(term_counts)
"""

Then, calculate the jargon distances:

```
j.calculate_jargon_distance()
j.write_to_file('jargon_distance.csv')
```
