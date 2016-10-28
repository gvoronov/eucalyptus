package tree

import datavalue._

sealed abstract class Node {
  val isLeaf
  // def getChild(): Node
  // def setChild(): Unit
  def predict(): DataValue
  def getLeaf(): Leaf
}

class BiNode(val feature: String, val split: Double, ) extends Node {
  var children: Map[Boolean, Node] = Map.empty

  val isLeaf = false

  def getChild(key: Boolean): Node = children.get(key).getOrElse(new BiNode())
  def setChild(key: Boolean, value: Node): Unit = {children(key) = value}
  def predict(row): DataValue = {
    val key: Boolean = row.select(column) match {
      case value: NumericalValue => value() >= split
      case value: SimpleCategoricalValue =>
      case value: ClassCategoricalValue =>
    }
  }
  def getLeaf(): Leaf
}

class Leaf extends Node {
  val isLeaf = true
  // def getChild(): Node
  // def setChild(): Unit
  def predict(): DataValue
  def getLeaf(): Leaf
}
