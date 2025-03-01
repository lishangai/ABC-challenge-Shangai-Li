Normal/Unusual Activity Recognition Challenge 2025

==========
Dataset Structure
Sensor Data:

- Main directory: users_timeXYZ/users/
- Contains multiple subdirectories with random numerical names (e.g., '38', '1716')
- Each subdirectory contains 1/multiple CSV files with accelerometer data
- File naming pattern: user-acc_[DIR-NUMBER]_[TIMESTAMP]_[RANDOM-NUMBER].csv

E.g., '38' folder has 5 files: 
	user-acc_38_2024-09-08T23_31_01.510+0100_97016.csv 
	user-acc_38_2024-09-08T23_31_16.519+0100_15638.csv
	... 

Sensor Data File Format:
Each CSV file contains 5 columns:

- Random identifier (to be ignored)
- Timestamp
- x-axis accelerometer data
- y-axis accelerometer data
- z-axis accelerometer data

==========
Activity Labels
File: TrainActivities.csv
Contains 7 columns:

- ID (random identifier)
- Activity Type ID
- Activity Type (10 distinct activity classes): You need to recognize them.
- Start Time
- End Time
- Update Time
- Subject ID (e.g., U1, U2, U3, ..., U21, U22)

E.g., 
ID	Activity Type ID	Activity Type						Started			Finished		Updated			Subject
1130251	2806			1 (FACING camera) Sit and stand				02/09/2024 06:16	02/09/2024 06:16	02/09/2024 06:16	U22
1130254	2807			2 (FACING camera) both hands SHAKING (sitting position)	02/09/2024 06:17	02/09/2024 06:17	02/09/2024 06:17	U22
1130257	2807			2 (FACING camera) both hands SHAKING (sitting position)	02/09/2024 06:18	02/09/2024 06:18	02/09/2024 06:18	U22
1130261	2806			1 (FACING camera) Sit and stand				02/09/2024 06:20	02/09/2024 06:20	02/09/2024 06:20	U22
1130292	2806			1 (FACING camera) Sit and stand				02/09/2024 06:42	02/09/2024 06:42	02/09/2024 06:42	U2
...
...

Activities are: 
1 (FACING camera) Sit and stand
2 (FACING camera) both hands SHAKING (sitting position)
3 Stand up from chair - both hands with SHAKING
4 (Sideway) Sit & stand
5 (Sideway) both hands SHAKING (sitting)
6 (Sideway) STAND up with - both hands SHAKING
7 Cool down - sitting/relax
8 Walk (LEFT --> Right --> Left)
9 Walk & STOP/frozen, full body shaking, rotate then return back
10 Slow walk (SHAKING hands/body, tiny step, head forward)

==========
Challenge Description
Objective: Develop a model to recognize 10 different activities based on accelerometer data.

Training Data: 
- 9 subjects provided for training. 9 subjects are: U1, U2, U3, U4, U5, U6, U7, U21, and U22.  
- Each subject performs all 10 activities unless any missing data.
- Most activities have multiple repetitions.
Note: Some time gaps may exist due to data collection issues

Key Tasks:
- Link accelerometer data with corresponding activities and subjects
- Develop recognition model(s) for the 10 activity classes
- Implement appropriate train/test splitting strategies
- Document your approach and results so that you can write a paper 
(Your paper will be published at the IJABC journal or IEEE Xplore - after acceptance and presentation at the conference in-person/online).

Important Notes:
- Test data will be provided separately (we will email you - so do not worry for this)
- Submission file format and complete code requirements will be announced later
- Final rankings will be announced during the conference
- Paper submissions should only present results based on the training data


All the best.   