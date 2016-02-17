"""Decision tree functions."""
import collections

NUMERIC = 'numeric'


def majority_value(data, target_attr, attributes_meta):
    """Return the majority label in data."""

    data = data[:]
    return most_frequent([record[target_attr] for record in data],
                         target_attr, attributes_meta)


def most_frequent(list_, target_attr, attributes_meta):
    """Return the item that appears most frequently in the given list.
    """

    list_ = list_[:]
    highest_freq = 0
    most_freq = None

    for val in unique(list_):
        val_count = list_.count(val)
        # breaking tie with first value listed in ARFF header
        if val_count == highest_freq:
            target_attr_vals = attributes_meta[target_attr]
            if target_attr_vals.index(val) < target_attr_vals.index(
                    most_freq):
                most_freq = val
        elif val_count > highest_freq:
            most_freq = val
            highest_freq = list_.count(val)

    return most_freq


def unique(list_):
    """Return the unique elements of the list_."""

    list_ = list_[:]
    unique_list = []

    for item in list_:
        if item not in unique_list:
            unique_list.append(item)

    return unique_list


def get_values(data, attr, attributes_meta):
    """Return the distinct values for a nominal feature. For a numeric
    feature, return the midpoint of its values."""

    data = data[:]
    attr_values = [record[attr] for record in data]
    attr_values.sort()

    if attributes_meta[attr] == NUMERIC:
        print '%s: %s' % (attr, attr_values)
        return (attr_values[0] + attr_values[-1]) / 2.0

    return unique(attr_values)


def pick_attribute(data, attributes, target_attr, heuristic, attributes_meta):
    """Pick the attribute with the most information gain."""

    data = data[:]
    # iii) stop when no feature has positive information gain
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gain = heuristic(data, attr, target_attr, attributes_meta)
        # If there is a tie between two features in their information
        # gain, you should break the tie in favor of the feature listed
        # first in the header section of the ARFF file
        if gain == best_gain and attr != target_attr:
            if (best_attr is None or attributes.index(attr) <
                    attributes.index(best_attr)):
                best_gain = gain
                best_attr = attr
        if gain > best_gain and attr != target_attr:
            best_gain = gain
            best_attr = attr

    return best_attr


def get_examples(data, attr, value, attributes_meta, lte=False):
    """Returns a list of all the records in data with the value of attr
    matching the given value in case of a nominal attribute and in case
    of numeric attributes, records with value of attr <= value if lte
    is True or values > otherwise."""

    data = data[:]
    ret_lst = []

    if not data:
        return ret_lst
    else:
        record = data.pop()
        if attributes_meta[attr] == NUMERIC:
            if lte:
                if record[attr] <= value:
                    ret_lst.append(record)
            else:
                if record[attr] > value:
                    ret_lst.append(record)
        else:
            if record[attr] == value:
                ret_lst.append(record)

        ret_lst.extend(get_examples(data, attr, value, attributes_meta, lte))

        return ret_lst


def get_classification(record, tree, attributes_meta):
    """This function recursively traverses the decision tree and returns a
    classification for the given record."""

    # If the current node is a string, then we've reached a leaf node and
    # we can return it as our answer
    #if type(tree) == type("string"):
    if isinstance(tree, str):
        return tree

    # Traverse the tree further until a leaf node is found.
    else:
        attr = tree.keys()[0]
        if attributes_meta[attr] == NUMERIC:
            if eval('%s%s' % (record[attr], tree[attr].keys()[0])):
                t = tree[attr][0]
            else:
                t = tree[attr][1]
        else:
            t = tree[attr][record[attr]]
        return get_classification(record, t, attributes_meta)


def classify(tree, data, attributes_meta):
    """Returns a list of classifications for each of the records in the
    data list as determined by the given decision tree."""

    data = data[:]
    classification = []

    for record in data:
        classification.append(get_classification(record, tree, attributes_meta))

    return classification


def create_decision_tree(data, attributes, attributes_meta, target_attr,
                         heuristic_func, min_instances):
    """Build a decision tree based on data."""

    data = data[:]
    labels = [record[target_attr] for record in data]  # get list of all labels
    default = majority_value(data, target_attr, attributes_meta)

    # iv) no remaining candidate splits at the node
    # minus 1 because of the target attribute
    if len(attributes) - 1 <= 0:
        return default
    # ii) fewer than m training instances reaching the node
    elif len(data) < min_instances:
        return default
    # i) all of the training instances reaching the node belong to the
    #    same class
    elif labels.count(labels[0]) == len(labels):
        return labels[0]
    else:
        # Choose the next best attribute to best classify our data
        best = pick_attribute(data, attributes, target_attr, heuristic_func,
                              attributes_meta)
        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object
        tree = {best: collections.defaultdict(lambda: default)}

        candidate_splits = get_values(data, best, attributes_meta)
        if attributes_meta[best] == NUMERIC:
            subtree = create_decision_tree(get_examples(data, best,
                                                        candidate_splits,
                                                        attributes_meta, True),
                                           [attr for attr in attributes
                                            if attr != best], attributes_meta,
                                           target_attr, heuristic_func,
                                           min_instances)
            tree[best]['<=%s' % candidate_splits] = subtree

            subtree = create_decision_tree(get_examples(data, best,
                                                        candidate_splits,
                                                        attributes_meta, False),
                                           [attr for attr in attributes
                                            if attr != best], attributes_meta,
                                           target_attr, heuristic_func,
                                           min_instances)
            tree[best]['>%s' % candidate_splits] = subtree
        else:
            # Create a new decision tree/sub-node for each of the values in the
            # best attribute field
            for val in candidate_splits:
                # Create a subtree for the current value under the "best" field
                subtree = create_decision_tree(
                    get_examples(data, best, val, attributes_meta),
                    [attr for attr in attributes if attr != best],
                    attributes_meta, target_attr, heuristic_func, min_instances)

                # Add the new subtree to the empty dictionary object in our new
                # tree/node we just created.
                tree[best][val] = subtree

    return tree
