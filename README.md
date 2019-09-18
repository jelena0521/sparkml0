#仅作为学习日记，不可用于任何商业用途
#仓库中包含三个python 第一个是分类 基于逻辑回归、SVM、随机森林和决策树 第二个是针对逻辑回归做参数优化 使用网格搜索 第三个是基于第二个减少参数
#第一个基本步骤
#1、定义标签，及标签类型 得到数据结构schema
#2、安装数据结构读取数据
#3、对读取的数据进行转变 主要是string转为int 并且onehot化
#4、将变量合成一个并命名为features
#5、定义模型，指定超参及labelcol
#6、定义pipeline 将3\4\5传入
#7、分割数据 randomsplit
#8、训练模型 pipeline.fit(births_train)
#9、测试模型 model.transform(births_test)
#10、评估模型 定义ev
#11、打印ROC
#12、保存模型  定义path model.write().overwrite().save(path)
#13、加载模型  用PipelineModel.load(path)
#第二个 
#1、在第五步定义模型时不要定义超参
#2、定义网格  grid
#3、定义网格交叉验证模型 cv
#4、将3、4传入pipeline
#5、训练 cv.fit(pipeline.fit(births_train).transform(births_train))
#6、测试  cvmodel.transform(pipeline.fit(births_train).transform(births_test))
#第三个
#1、在第4步下加一个select=ft.ChiSqSelector
