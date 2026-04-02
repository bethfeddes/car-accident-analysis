# US Accident Severity Analysis (Illinois) 
Applies the FP-Growth association rule mining algorithm to a 500,000-record US car accident sample dataset, filtered to Illinois, to discover patterns between environmental conditions and accident severity.

## Dataset
[US Accidents (March 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) by Sobhan Moosavi- 500k sample version. Place the CSV in your ~/Downloads folder before running.
## Requirements  
- Python 3
- pandas
- mlxtend
## How to Run
python analysis.py
  Output prints the top association rules for Severity 2 and Severity 3-4 accidents, formatted with support, confidence, and lift, sorted by strongest lift.
## Key Findings
- The most severe crashes (Severity 3-4) are strongly associated with nighttime conditions
- Presence of both a traffic signal and stop sign is the strongest predictor of Severity 2 accidents (confidence: 94.9%, lift: 1.53)
- Low visibility combined with traffic signals or crossings is a notable factor in Severity 2 crashes
- Severity 1 and 4 were too underrepresented in Illinois data to produce rules under the thresholds used
## Configuration
Thresholds can be adjusted directly in the script:
- Minimum support: 0.003
- Minimum confidence: 0.4
- Minimum lift: 1.5
