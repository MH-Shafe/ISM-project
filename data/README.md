# CERT Insider Threat Dataset r4.2

This directory holds the CERT r4.2 dataset used by the project.

## Expected structure

```
data/raw/cert_r4.2/
├── r4.2/
│   ├── logon.csv
│   ├── device.csv
│   ├── http.csv
│   ├── file.csv
│   ├── email.csv
│   ├── ldap.csv
│   ├── ps.csv
│   └── ...
└── answers/
    └── insiders.csv
```

## How to obtain

1. Download CERT r4.2 from the SEI CERT website:
   https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099

2. Extract the archive.

3. Place the extracted files under `data/raw/cert_r4.2/` preserving the directory structure above.

## Important

- Do NOT substitute another CERT release (r6.2, r4.1, etc.).
- Do NOT commit dataset files to Git (they are excluded by `.gitignore`).
- The dataset is large (~1.5 GB compressed).
