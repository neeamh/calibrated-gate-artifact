# Source data

The campaign uses the public UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset
(Clore, Cios, DeShazo and Strack, 2014; CC BY 4.0):
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The data are not redistributed here. To rerun the campaign, download the archive, extract
`diabetic_data.csv` into this folder, and point `data.raw_csv` in the notebook's configuration
cell to `data/diabetic_data.csv`. The cohort rule (earliest encounter per patient, 30-day
readmission outcome, 40/40/20 stratified split) is applied by the campaign code.

Regenerating the paper's numbers does not need this file: it uses the saved results in
`results/main_v5/`.
