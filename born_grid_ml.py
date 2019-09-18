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
#选择特征值
selector=ft.ChiSqSelector(numTopFeatures=6,featuresCol=featuresCreator.getOutputCol(),outputCol='selectedFeatures',labelCol="INFANT_ALIVE_AT_REPORT")
#分割数据
births_train,births_test=births.randomSplit([0.7,0.3],seed=666)
#建立分类模型
#逻辑回归
logistic=cl.LogisticRegression(labelCol='INFANT_ALIVE_AT_REPORT',featuresCol='selectedFeatures')
#创建网格
grid=tune.ParamGridBuilder().addGrid(logistic.maxIter,[2,10,50]).addGrid(logistic.regParam,[0.01,0.05,0.03]).build()
#创建pipeline
pipeline=Pipeline(stages=[encoder,featuresCreator,selector])
#指定评估
evaluator=ev.BinaryClassificationEvaluator(rawPredictionCol='probability',labelCol="INFANT_ALIVE_AT_REPORT")
#训练cv模型
cv=tune.CrossValidator(estimator=logistic,estimatorParamMaps=grid,evaluator=evaluator)
cvmodel=cv.fit(pipeline.fit(births_train).transform(births_train))
#测试模型
test_model=cvmodel.transform(pipeline.fit(births_train).transform(births_test))
print(evaluator.evaluate(test_model,{evaluator.metricName:'areaUnderROC'}))
print(evaluator.evaluate(test_model,{evaluator.metricName:'areaUnderPR'}))
