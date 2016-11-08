package eucalyptus.tree
/**
 * Provides classes for implementing a tree structure that is required for implementing a
 * decision tree.
 */

import scala.collection.mutable.{Map => MutableMap}

import koalas.numericalops.NumericalOpsImps._
import koalas.datavalue._
import koalas.row.Row
import koalas.series.Series
import koalas.dataframe.DataFrame
import koalas.series.Series

sealed abstract class Node {
  val isLeaf: Boolean

  def getChild(key: Boolean): Node
  def getChild(row: Row): Node
  def setChild(key: Boolean, node: Node): Unit
  def setAndGetChild(key: Boolean, node: Node): Node

  def predict[T](row: Row): T
  def getLeaf(row: Row): Leaf
}

/**
 * [feature description]
 * @constructor creates a new BiNode that will contain two children keyed by a boolean type.
 * @param feature the specific predicot that the BiNode uses it it's split rule
 * @param split the value the BiNode uses for it's split rule
 * @param catMap optional parameter that is used to map CategoricalValue predictors to
 *        NumericalValue so that a split rule may be applied.
 */
class BiNode(
    val feature: String, val split: Double, val catMap: Map[DataValue, NumericalValue]=Map.empty)
    extends Node {
  protected var children: MutableMap[Boolean, Node] = MutableMap.empty

  /** A BiNode is explicitly not a leaf */
  val isLeaf = false

  /**
   * Get child node that is specified by key parameter, no need to apply split rule here.
   * @param key
   */
  def getChild(key: Boolean): Node = {
    children
      .get(key)
      .getOrElse(throw new RuntimeException(
        "This tree is malformed, it contains leafless branches"))
  }

  /**
   * Get appropriate child node by applying split rule to row.
   * @param row
   */
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
  def setAndGetChild(key: Boolean, node: Node): Node = {
    setChild(key, node)
    getChild(key)
  }

  def predict[T](row: Row): T = getChild(row).predict[T](row)
  def getLeaf(row: Row): Leaf = getChild(row).getLeaf(row)
}

abstract class Leaf(responseSeries: Series[DataValue]) extends Node {
  val prediction: DataValue

  val isLeaf: Boolean = true
  val numPoints: Int = responseSeries.length



  def getChild(key: Boolean): Node = throw new RuntimeException(
    "method getChild not defined for Leaf Node type!")
  def getChild(row: Row): Node = throw new RuntimeException(
    "method getChild not defined for Leaf Node type!")
  def setChild(key: Boolean, node: Node): Unit = throw new RuntimeException(
    "method setChild not defined for Leaf Node type!")
  def setAndGetChild(key: Boolean, node: Node): Node = throw new RuntimeException(
    "method setAndGetChild not defined for Leaf Node type!")

  def predict[T](row: Row): T = prediction.asInstanceOf[T]
  def getLeaf(row: Row): Leaf = this
}

class RegressionLeaf(responseSeries: Series[NumericalValue])
    extends Leaf(responseSeries) {
  // Insert check that DataValue is really NumericalValue
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


class BiTree(val root: Node) {
  val isLeaf: Boolean = root match {
    case _: BiNode => false
    case _: Leaf => true
    case _ => throw new RuntimeException("parameter root is not of type BiNode or Leaf!")
  }

  def getRoot: Node = root

  def predict[T](row: Row): T = root.predict[T](row)
  def predict[T](df: DataFrame): Series[T] = df.map((row: Row) => predict[T](row))
  def getLeaf(row: Row): Leaf = root.getLeaf(row)
}
