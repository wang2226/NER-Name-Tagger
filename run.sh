# javac -cp maxent-3.0.0.jar:trove.jar:. MEtrain.java
# javac -cp maxent-3.0.0.jar:trove.jar:. MEtag.java

python FeatureBuilder.py
java -cp maxent-3.0.0.jar:trove.jar:. MEtrain feature-enhanced-training model

java -cp maxent-3.0.0.jar:trove.jar:. MEtag feature-enhanced-dev model dev.result
java -cp maxent-3.0.0.jar:trove.jar:. MEtag feature-enhanced-test model test.name

python score.name.py ./NAME_CORPUS_FOR_STUDENTS/dev.name dev.result 

