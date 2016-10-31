package eucalyptus.tree

import scala.collection.mutable.{Map => MutableMap}

import koalas.datavalue._
import koalas.row.Row
import koalas.series.Series

sealed abstract class Node {
  val isLeaf: Boolean
  // def getChild(): Node
  // def setChild(): Unit
  def predict(row: Row): DataValue
  def getLeaf(row: Row): Leaf
}

class BiNode(
    val feature: String, val split: Double, val catMap: Map[DataValue, NumericalValue]=Map.empty)
    extends Node {
  protected var children: MutableMap[Boolean, Node] = MutableMap.empty

  val isLeaf = false

  def getChild(key: Boolean): Node = {
    children
      .get(key)
      .getOrElse(throw new RuntimeException(
        "This tree is malformed, it contains leafless branches"))
  }
  def getChild(row: Row): Node = {
    val key: Boolean = row.select(feature) match {
      case value: NumericalValue => value >= split
      case value: CategoricalValue => catMap(value) >= split
      case _ => throw new RuntimeException(
        "feature " + feature + " of row contains value that can't be evaluate")
    }
    getChild(key)
  }
  def setChild(key: Boolean, node: Node): Unit = {children(key) = node}

  def predict(row: Row): DataValue = getChild(row).predict(row)
  def getLeaf(row: Row): Leaf = getChild(row).getLeaf(row)
}

abstract class Leaf(responseSeries: Series[DataValue]) extends Node {
  val prediction: DataValue

  val isLeaf: Boolean = true
  val numPoints: Int = responseSeries.length
  // def getChild(): Node
  // def setChild(): Unit
  def predict(row: Row): DataValue = prediction
  def getLeaf(row: Row): Leaf = this
}

class RegressionLeaf(val responseSeries: Series[DataValue])
    extends Leaf(responseSeries) {
  val mean: NumericalValue = responseSeries.mean
  val variance: NumericalValue = responseSeries.variance

  override val prediction: NumericalValue = mean
}

// class ClassificationLeaf(val responseSeries: Series[CategoricalValue])
//     extends Leaf[CategoricalValue](responseSeries) {
  // val classCounts
  // val classProbabilities
  //
  // override val prediction: CategoricalValue
// }
