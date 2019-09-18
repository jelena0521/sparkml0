import pyspark.sql.types as typ
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('spark0').getOrCreate()
import pyspark.ml.feature as ft
import pyspark.ml.classification as cl
import pyspark.ml.regression as reg
import pyspark.ml.clustering as clu
from  pyspark.ml import Pipeline,PipelineModel
import pyspark.ml.evaluation as ev
import pyspark.ml.tuning as tune
#定义数据结构
labels = [
    ('INFANT_ALIVE_AT_REPORT', typ.IntegerType()),
    ('BIRTH_PLACE', typ.StringType()),
    ('MOTHER_AGE_YEARS', typ.IntegerType()),
    ('FATHER_COMBINED_AGE', typ.IntegerType()),
    ('CIG_BEFORE', typ.IntegerType()),
    ('CIG_1_TRI', typ.IntegerType()),
    ('CIG_2_TRI', typ.IntegerType()),
    ('CIG_3_TRI', typ.IntegerType()),
    ('MOTHER_HEIGHT_IN', typ.IntegerType()),
    ('MOTHER_PRE_WEIGHT', typ.IntegerType()),
    ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
    ('MOTHER_WEIGHT_GAIN', typ.IntegerType()),
    ('DIABETES_PRE', typ.IntegerType()),
    ('DIABETES_GEST', typ.IntegerType()),
    ('HYP_TENS_PRE', typ.IntegerType()),
    ('HYP_TENS_GEST', typ.IntegerType()),
    ('PREV_BIRTH_PRETERM', typ.IntegerType())]

schema=typ.StructType([typ.StructField(e[0],e[1],False) for e in labels])
#读入数据
births=spark.read.csv('births_transformed.csv',header=True,schema=schema)
#string转为int
births=births.withColumn('BIRTH_PLACE_INT',births['BIRTH_PLACE'].cast(typ.IntegerType()))
births=births.withColumn('INFANT_ALIVE_AT_REPORT',births['INFANT_ALIVE_AT_REPORT'].cast(typ.DoubleType()))
#转为one-hot
encoder=ft.OneHotEncoder(inputCol='BIRTH_PLACE_INT',outputCol='BIRTH_PLACE_VEC')
#将所有的变量合并
featuresCreator=ft.VectorAssembler(inputCols=[col[0] for col in labels[2:]]+[encoder.getOutputCol()],outputCol='features')
#分割数据
births_train,births_test=births.randomSplit([0.7,0.3],seed=666)
#建立分类模型
#逻辑回归
logistic=cl.LogisticRegression(maxIter=10,regParam=0.1,labelCol='INFANT_ALIVE_AT_REPORT')
#随机森林
randomforest=cl.RandomForestClassifier(numTrees=5,maxDepth=10,labelCol='INFANT_ALIVE_AT_REPORT')
#决策树
decisiontree=cl.DecisionTreeClassifier(maxDepth=10,labelCol='INFANT_ALIVE_AT_REPORT')
#svm
svm=cl.LinearSVC(maxIter=10,regParam=0.1,labelCol='INFANT_ALIVE_AT_REPORT')
#创建pipeline
pipeline=Pipeline(stages=[encoder,featuresCreator,logistic])
#训练数据
model=pipeline.fit(births_train)
#测试
test_model=model.transform(births_test)
#评估
evaluator=ev.BinaryClassificationEvaluator(rawPredictionCol='probability',labelCol="INFANT_ALIVE_AT_REPORT")
print(evaluator.evaluate(test_model,{evaluator.metricName:'areaUnderROC'}))
print(evaluator.evaluate(test_model,{evaluator.metricName:'areaUnderPR'}))
#保存模型
modelpath='logistic_model'
model.write().overwrite().save(modelpath)
#加载模型
loadmodel=PipelineModel.load(modelpath)
load_test_model=loadmodel.transform(births_test)
