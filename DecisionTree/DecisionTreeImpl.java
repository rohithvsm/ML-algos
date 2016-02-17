import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 * 
 * You must add code for the 5 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
	private DecTreeNode root;
	private List<String> labels; // ordered list of class labels
	private List<String> attributes; // ordered list of attributes
	private Map<String, List<String>> attributeValues; // map to ordered
														// discrete values taken
														// by attributes
	private List<AccuracyInfo> prunTree = new ArrayList<AccuracyInfo>();

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary
		// this is void purposefully
	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train
	 *            : the training set
	 */
	DecisionTreeImpl(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: add code here

		// My code starts here
		int labelCount[] = new int[this.labels.size()];
		// instances
		for (int i = 0; i < train.instances.size(); i++) {
			// labels
			for (int j = 0; j < this.labels.size(); j++) {
				if (train.instances.get(i).label == j) {
					labelCount[j]++;
				}
			}
		}
		int max = 0;
		for (int i = 0; i < this.labels.size(); i++) {
			if (labelCount[i] > labelCount[max])
				max = i;
		}

		List<Integer> attrs = new ArrayList<Integer>();
		for (int i = 0; i < this.attributes.size(); i++)
			attrs.add(i);

		// call the algorithm
		this.root = decisionTreeLearning(train.instances, attrs, max, -1);
	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning
	 * set.
	 * 
	 * @param train
	 *            : the training set
	 * @param tune
	 *            : the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: add code here

		int numInstances = train.instances.size();
		int lbl[] = new int[this.labels.size()];
		int max = 0, i, j = 0;
		for (i = 0; i < numInstances; i++) {
			for (j = 0; j < this.labels.size(); j++) {
				if (train.instances.get(i).label == j) {
					lbl[j]++;
					if (lbl[j] > lbl[max])
						max = j;
				}
			}
		}

		List<Integer> attrbts = new ArrayList<Integer>();
		for (i = 0; i < (this.attributes.size()); i++)
			attrbts.add(i);
		this.root = decisionTreeLearning(train.instances, attrbts, max, -1);
		double acc = calculateAccuracy(tune.instances);
		double currMaxAcc = 0.0, prevMaxAcc = acc;
		int maxAccIdx = 0;
		int depth, minDepth;

		do {
			depth = 0;
			minDepth = Integer.MAX_VALUE;
			currMaxAcc = 0;
			pruneTree(this.root, tune.instances, depth);
			for (i = 0; i < prunTree.size(); i++) {
				if (prunTree.get(i).accuracy == currMaxAcc) {
					if (prunTree.get(i).depth < minDepth) {
						currMaxAcc = prunTree.get(i).accuracy;
						maxAccIdx = i;
						minDepth = prunTree.get(i).depth;
					}
				} else if (prunTree.get(i).accuracy > currMaxAcc) {
					currMaxAcc = prunTree.get(i).accuracy;
					maxAccIdx = i;
					minDepth = prunTree.get(i).depth;
				}
			}
			if (currMaxAcc >= prevMaxAcc) {
				root = prunTree.get(maxAccIdx).rootTree;
				prevMaxAcc = currMaxAcc;
			}
			prunTree.clear();
		} while (currMaxAcc >= acc);
	}

	private double entropyAttr(List<Instance> instances, Integer attrIdx) {
		List<String> attrValues = new ArrayList<String>(
				this.attributeValues.get(this.attributes.get(attrIdx)));
		int numInstancesHavingAttrVal[][] = new int[this.labels.size()][attrValues
				.size()];
		int numInstances = instances.size();
		double entropy = 0.0, probablty;
		int lbl[] = new int[attrValues.size()];
		for (int i = 0; i < numInstances; i++) {
			for (int j = 0; j < this.labels.size(); j++) {
				if (instances.get(i).label == j) {
					numInstancesHavingAttrVal[j][instances.get(i).attributes
							.get(attrIdx)]++;
					lbl[instances.get(i).attributes.get(attrIdx)]++;
				}
			}
		}
		double netEntropy = 0.0;
		for (int i = 0; i < attrValues.size(); i++) {
			entropy = 0.0;
			for (int j = 0; j < this.labels.size(); j++) {
				probablty = ((double) numInstancesHavingAttrVal[j][i])
						/ (lbl[i]);
				if (probablty > 0 && probablty < 1)
					entropy = entropy + (-1) * probablty
							* (Math.log(probablty) / Math.log(2));
			}
			probablty = ((double) lbl[i] / numInstances);
			netEntropy = netEntropy + probablty * entropy;
		}
		return netEntropy;
	}

	private double labelEntropy(List<Instance> instances) {
		int labelCount[] = new int[this.labels.size()];
		for (int i = 0; i < instances.size(); i++)
			labelCount[instances.get(i).label]++;
		int numInstances = instances.size();
		double entropy = 0.0, probablty;
		int lbl[] = new int[this.labels.size()];
		for (int i = 0; i < numInstances; i++) {
			for (int j = 0; j < this.labels.size(); j++) {
				if (instances.get(i).label == j)
					lbl[j]++;
			}
		}
		for (int i = 0; i < this.labels.size(); i++) {
			probablty = ((double) lbl[i] / numInstances);
			if (probablty != 0 && probablty != 1)
				entropy = entropy + (-1) * probablty
						* (Math.log(probablty) / Math.log(2));
		}
		return entropy;
	}

	private DecTreeNode decisionTreeLearning(List<Instance> instances,
			List<Integer> attrs, Integer majLabel, Integer parentattributeValue) {
		int i = 0, j = 0;
		double entropyLabels = labelEntropy(instances);

		int instancesPerLabel[] = new int[this.labels.size()];
		int max = 0, majorityLabel = Integer.MAX_VALUE;
		for (i = 0; i < instances.size(); i++) {
			instancesPerLabel[instances.get(i).label]++;
			if (instancesPerLabel[instances.get(i).label] == max) {
				if (instances.get(i).label < majorityLabel) {
					max = instancesPerLabel[instances.get(i).label];
					majorityLabel = instances.get(i).label;
				}
			} else if (instancesPerLabel[instances.get(i).label] > max) {
				max = instancesPerLabel[instances.get(i).label];
				majorityLabel = instances.get(i).label;
			}
		}

		if (instances.size() == 0) {
			DecTreeNode node = new DecTreeNode(majLabel, Integer.MAX_VALUE,
					parentattributeValue, true);
			return node;
		} else if (attrs.size() == 0) {
			DecTreeNode node = new DecTreeNode(majorityLabel,
					Integer.MAX_VALUE, parentattributeValue, true);
			return node;
		} else if (entropyLabels == 0) {
			Instance example = instances.get(0);
			DecTreeNode node = new DecTreeNode(example.label,
					Integer.MAX_VALUE, parentattributeValue, true);
			return node;
		} else {
			double attrEntropy[] = new double[attrs.size()];
			double infoGain[] = new double[attrs.size()];
			double maxInfoGain = 0.0;
			int attrMaxInfoGain = Integer.MAX_VALUE;

			for (i = 0; i < attrs.size(); i++) {
				attrEntropy[i] = entropyAttr(instances, attrs.get(i));
				infoGain[i] = entropyLabels - attrEntropy[i];
				if (infoGain[i] == maxInfoGain) {
					if (i < attrMaxInfoGain) {
						attrMaxInfoGain = i;
						maxInfoGain = infoGain[i];
					}
				} else if (infoGain[i] > maxInfoGain) {
					attrMaxInfoGain = i;
					maxInfoGain = infoGain[i];
				}
			}
			int numChildren;
			int actualAttr;
			String attrName;
			actualAttr = attrs.get(attrMaxInfoGain);
			attrName = this.attributes.get(actualAttr);

			numChildren = (this.attributeValues.get(attrName)).size();

			DecTreeNode node = new DecTreeNode(majorityLabel, actualAttr,
					parentattributeValue, false);

			List<List<Instance>> newInstances = new ArrayList<List<Instance>>();
			List<Integer> newAttrs = new ArrayList<Integer>(attrs);

			newAttrs.remove(attrMaxInfoGain);

			for (i = 0; i < instances.size(); i++) {
				Instance currInstance = instances.get(i);
				for (j = 0; j < numChildren; j++) {
					if (i == 0)
						newInstances.add(new ArrayList<Instance>());
					if ((currInstance.attributes).get(actualAttr) == j)
						newInstances.get(j).add(currInstance);
				}
			}

			List<DecTreeNode> children = new ArrayList<DecTreeNode>(numChildren);

			for (i = 0; i < numChildren; i++) {
				DecTreeNode child = decisionTreeLearning(newInstances.get(i),
						newAttrs, majorityLabel, i);
				children.add(child);
				node.children = children;
			}
			return node;
		}
	}

	private DecTreeNode copyNode(DecTreeNode node) {
		DecTreeNode copy = new DecTreeNode(node.label, node.attribute,
				node.parentAttributeValue, node.terminal);
		if (!node.terminal)
			copy.children = copyList(node.children);
		return copy;
	}

	private List<DecTreeNode> copyList(List<DecTreeNode> list) {
		int i = 0;
		List<DecTreeNode> copylist = new ArrayList<DecTreeNode>(list.size());
		for (i = 0; i < list.size(); i++) {
			DecTreeNode child = copyNode(list.get(i));
			copylist.add(child);
		}

		return copylist;
	}

	private double calculateAccuracy(List<Instance> instancesSet) {
		int i = 0, j = 0;
		int positiveCount = 0, negativeCount = 0;
		for (i = 0; i < instancesSet.size(); i++) {
			int labelIndex = this.labels.indexOf(classify(instancesSet.get(i)));
			if (labelIndex == (instancesSet.get(i).label))
				positiveCount++;
			else
				negativeCount++;
		}
		double accuracy = ((double) (positiveCount)) / (instancesSet.size());
		return accuracy;
	}

	private void pruneTree(DecTreeNode rootNode, List<Instance> tuneSet,
			int depth) {

		if (!rootNode.terminal) {
			int i = 0;
			List<DecTreeNode> childrenList = copyList(rootNode.children);
			rootNode.children = null;
			rootNode.terminal = true;

			double accuracy = calculateAccuracy(tuneSet);

			AccuracyInfo accInfo = new AccuracyInfo();
			accInfo.rootTree = copyNode(root);
			accInfo.accuracy = accuracy;
			accInfo.depth = depth;

			this.prunTree.add(accInfo);

			rootNode.terminal = false;
			rootNode.children = new ArrayList<DecTreeNode>();
			rootNode.children = copyList(childrenList);
			depth++;
			for (i = 0; i < (rootNode.children.size()); i++)
				pruneTree(rootNode.children.get(i), tuneSet, depth);

		}

	}

	@Override
	public String classify(Instance instance) {

		// TODO: add code here

		DecTreeNode node = new DecTreeNode(root.label, root.attribute,
				root.parentAttributeValue, root.terminal);
		if (!root.terminal)
			node.children = copyList(root.children);
		int attr;
		int attrVal;
		while (!node.terminal) {
			attr = node.attribute;
			attrVal = instance.attributes.get(attr);
			node = node.children.get(attrVal);
		}
		return this.labels.get(node.label);
	}

	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {

		printTreeNode(root, null, 0);
	}

	/**
	 * Prints the subtree of the node with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else {
			String parentAttribute = attributes.get(parent.attribute);
			value = attributeValues.get(parentAttribute).get(
					p.parentAttributeValue);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + labels.get(p.label) + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + attributes.get(p.attribute) + "?}");
			System.out.println(sb.toString());
			for (DecTreeNode child : p.children) {
				printTreeNode(child, p, k + 1);
			}
		}
	}

	@Override
	public void rootInfoGain(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: add code here

		double entropyLabels = labelEntropy(train.instances);
		double attrEntropy[] = new double[this.attributes.size()];
		double infoGain[] = new double[this.attributes.size()];
		double maxInfoGain = 0.0;
		int attrMaxInfoGain = 0;

		for (int i = 0; i < attributes.size(); i++) {
			attrEntropy[i] = entropyAttr(train.instances, i);
			infoGain[i] = entropyLabels - attrEntropy[i];
			System.out.print(this.attributes.get(i) + " ");
			System.out.format("%.5f\n", infoGain[i]);
		}
	}
}

class AccuracyInfo {
	DecTreeNode rootTree;
	public double accuracy;
	public int depth;
}
