# Data layout

Each dataset directory contains the exact fixed split used by the paper:

```text
data/<dataset>/train.csv
data/<dataset>/val.csv
data/<dataset>/test.csv
data/<dataset>/profiles/user_profiles.npy
data/<dataset>/profiles/item_profiles.npy
```

CSV columns are `user_id`, `item_id`, `ratings`, and `reviews`. IDs are zero-based and already remapped.

The source datasets are the Amazon 5-core review datasets associated with McAuley, Pandey, and Leskovec (KDD 2015). Review the upstream terms before redistributing the CSV files.
