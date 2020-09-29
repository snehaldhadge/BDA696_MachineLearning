import pyspark.sql.functions as func
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.window import Window


class BattingAvgCalc(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(BattingAvgCalc, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()

        # function to calculate number of seconds from number of days
        def days(i):
            return i * 86400

        windowSpec = (
            Window.partitionBy(dataset["batter"])
            .orderBy(dataset["l_dt"].cast("long"))
            .rangeBetween(days(-100), days(-1))
        )

        dataset = dataset.withColumn(
            output_col,
            func.coalesce(
                func.sum(dataset[input_cols[0]]).over(windowSpec)
                / func.sum(dataset[input_cols[1]]).over(windowSpec),
                func.lit(0.0),
            ),
        )
        # dataset = dataset.withColumn(
        #     output_col, dataset[input_cols[0]] / dataset[input_cols[1]]
        # )
        return dataset
