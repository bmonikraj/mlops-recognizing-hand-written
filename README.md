# mlops-recognizing-hand-written

## Assignment 4

| run   |        svm |   decision_tree |
|:------|-----------:|----------------:|
| 1     | 0.98441    |       0.815145  |
| 2     | 0.983287   |       0.844011  |
| 3     | 0.985185   |       0.866667  |
| 4     | 0.977654   |       0.837989  |
| 5     | 0.988764   |       0.876404  |
| mean  | 0.98386    |       0.848043  |
| std   | 0.00329028 |       0.0197942 |

## Flask app run 

`FLASK_APP=api.py flask run`

## Docker 

### Build 
`docker build -t exp:v1 -f Dockerfile .`

### Run 
`docker run -it exp:v1`