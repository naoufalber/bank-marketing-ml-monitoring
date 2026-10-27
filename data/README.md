# Dataset

**Bank Marketing**, Moro, S., Rita, P. & Cortez, P. (2014), UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5K306.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

The required `bank-additional-full.csv` contains 41,188 rows, 20 input columns and the binary target `y`. The verified run used the unchanged file in the supplied project archive. Run `python src/download_data.py` to obtain the same filename from UCI. Raw data is ignored by Git. The dataset is available under **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/ . `bank-additional-names.txt` contains the accompanying source documentation.

The default CLI excludes `duration`: it is only known after a call ends. Other feature availability still depends on when a real scoring decision would be made.
