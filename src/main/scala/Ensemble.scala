package eucalyptus.ensemble

import eucalyptus.decisiontree._

sealed trait Bagging {
  def fit (
      x: DataFrame, y: Option[Series[DataValue]] = None, predictors: Option[List[String]] = None,
      response: Option[String] = None, weights: Option[Series[NumericalValue]] = None): Unit = {

  }
  def predictEnsemble(x)

}

class BaggingRegressor(
    val numTrees: Int = 10, val maxSplitPoints: Int = 10, val minSplitPoints: Int = 1,
    val maxDepth: Int = 100, val minSamplesSplit: Int = 2, val minSamplesLeaf: Int = 1)
    extends Bagging {
  val forest = List.fill(numTrees)(
    new RegressionTree(maxSplitPoints, minSplitPoints, maxDepth, minSamplesSplit, minSamplesLeaf))  
}
class BaggingClassifier extends Bagging {
  throw new RuntimeException("Not implemented yet!")
}
class BaggingBivariateClassifier extends BaggingClassifier {
  throw new RuntimeException("Not implemented yet!")
}
class BaggingMultivariateClassifier extends BaggingClassifier {
  throw new RuntimeException("Not implemented yet!")
}
trait RandomForest
class RandomForestRegressor  extends BaggingRegressor with RandomForest {
  override val forest =
}
class RandomForestClassifier extends BaggingClassifier with RandomForest {
  throw new RuntimeException("Not implemented yet!")
}
class RandomForestBivariateClassifier extends RandomForestClassifier {
  throw new RuntimeException("Not implemented yet!")
}
class RandomForestMultivariateClassifier extends RandomForestClassifier {
  throw new RuntimeException("Not implemented yet!")
}
trait Boosting
trait AdaBoost
trait GradientBoosting
