# Wine Quality Prediction

## Links

* [Github](https://github.com/yash3012/cs643ProgAssignment2)

* [Docker](https://hub.docker.com/repository/docker/yash3012/winequality)

## 1. Parallel training implementation
* Create a cluster

We create a cluster of **4 nodes to train a ML model for predicting the quality of the wine using AWS EMR tool.

* Upload the files to s3 bucket.

**How to create a Cluster**

***Step i:*** Click `EMR` in dashboard under `Analytics` section 
***Step ii:*** Click `Create Cluster` 
***Step iii:*** Type desired cluster name in the `General Configuration` for `Cluster Name` .
***Step iii-a**: Under ``Software configuration` in the application column click the button which shows `Spark: Spark 2.4.7 on Hadoop 2.10.1 YARN and Zeppelin 0.8.2``.  
***Step iii-b:** Under `Hardware Configuration` click `m4.large` rather than the default `m5.xlarge` as the default m5.xlarge incurs a cost of $0.043/hr in contrast to the $0.03 for m4.large. Keep in mind that EMR incurs an additional 25% cost post first usage. 
***Step iii-c:** Select `4` instances under the column `Number of instances` 
**Step iii.d:** Under `Security and access` click the EC2 key pair if already created else create a new one
***Step iv:*** Click Create Cluster button. Wait for around 15 minutes for the cluster to start functioning. 

Now go to the `Master` instance and pull the files from s3 bucket.

```aws s3 cp s3://s3mywineproject/ ./ --recursive```

These files must be made available to the slave nodes.
```
hadoop fs -put TrainingDataset.csv
hadoop fs -put ValidationDataset.csv
```
We can confirm the accessibility of the files using
Now we can start training. We use `spark-submit` command to run the jar file.```spark-submit train.py```

A `model/ModelTrained` folder will be created having store trained model to it.
Verify model will be created when the following is executed:

Copy This folder back to our master node using following```hdfs dfs -copyToLocal model/ModelTrained /home/hadoop/wine_quality```

Now we can start training. We use `spark-submit` command to run the jar file. ``` spark-submit test.py ```

## 3. Docker container for prediction

Now we're going to build our docker image & create a container out of it.
Make sure you are login to docker.io on your local machine
Install the latest docker and start it.

Start the docker

Add ec2 instance to the docker

push the image to the docker registry(docker push yrp24/wine_quality:tagname).
Pull image from docker repository(docker pull yrp24/wine_quality:tagname).
