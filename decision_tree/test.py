import json
import pprint
from dtree import *
from id3 import *
import sys
import os.path

NUMERIC = 'numeric'


def parse_cmd_line_args():
    """Parse command-line arguments.

    dt-learn <train-set-file> <test-set-file> m

    """

    if len(sys.argv) < 4:
        print >> sys.stderr, 'Argument error: Not enough arguments'
        sys.exit(1)

    training_filename = sys.argv[1]
    test_filename = sys.argv[2]
    min_instances = int(sys.argv[3])

    if (not os.path.isfile(training_filename) or
            not os.path.isfile(test_filename)):
        print >> sys.stderr, 'Argument error: Illegal arguments'
        sys.exit(1)

    return training_filename, test_filename, min_instances


def get_attributes(filename):
    """Parses the attributes from the header of the ARFF file."""

    attributes = []
    attributes_meta = {}

    with open(filename) as fin:
        for line in fin:
            if line.startswith('%'):
                continue
            elif line.startswith('@'):
                if line.startswith('@data'):
                    break
                elif line.startswith('@attribute'):
                    line = line.strip().lstrip('@attribute').strip().lstrip("'")
                    line_parts = map(str.strip, line.split("'"))
                    attribute_name = line_parts[0]
                    attributes.append(attribute_name)
                    if not line_parts[1].startswith('{'):
                        attributes_meta[attribute_name] = NUMERIC
                    else:
                        attributes_meta[attribute_name] = (map(
                            str.strip, line_parts[1].lstrip('{').rstrip('}').
                                split(',')))
                else:  # skip relation
                    continue

    return attributes, attributes_meta


def get_data(filename, attributes, attributes_meta):
    """Given a data file and its attributes, return a list of dicts
    which represents the data in the file."""

    data = []

    # Convert the numeric attributes from string to numbers
    numeric_attrs = []
    for attr in attributes_meta:
        if attributes_meta[attr] == NUMERIC:
            numeric_attrs.append(attr)
    with open(filename) as fin:
        for line in fin:
            if line.startswith('%'): continue
            if line.startswith('@'): continue
            instance_dict = dict(zip(attributes,
                                     [datum.strip() for datum in line.strip().
                                     split(',')]))
            for numeric_attr in numeric_attrs:
                val = instance_dict[numeric_attr]
                if '.' in val:
                    instance_dict[numeric_attr] = float(val)
                else:
                    instance_dict[numeric_attr] = int(val)
            data.append(instance_dict)
            # data.append(dict(zip(attributes,
            #                     [datum.strip() for datum in line.strip().
            #                     split(',')])))

    return data


def print_tree(tree, str):
    """This function recursively crawls through the d-tree and prints it out in a
    more readable format than a straight print of the Python dict object."""
    if type(tree) == dict:
        print "%s%s" % (str, tree.keys()[0])
        for item in tree.values()[0].keys():
            print "%s\t%s" % (str, item)
            print_tree(tree.values()[0][item], str + "\t")
    else:
        print "%s\t->\t%s" % (str, tree)


def main():
    # Get training and test data from command-line arguments
    training_filename, test_filename, min_instances = parse_cmd_line_args()

    attributes = []
    attributes_meta = {}
    # Get the attributes and label
    attributes, attributes_meta = get_attributes(training_filename)
    target_attr = attributes[-1]

    # Get the training and test data
    training_data = get_data(training_filename, attributes, attributes_meta)
    test_data = get_data(test_filename, attributes, attributes_meta)

    # Build the decision tree
    dtree = create_decision_tree(training_data, attributes, attributes_meta,
                                 target_attr, gain, min_instances)

    #print json.dumps(dtree, indent=4)
    # Classify the records in the test data
    classification = classify(dtree, test_data, attributes_meta)

    # Print the results of the test
    print '------------------------\n'
    print '--   Classification   --\n'
    print '------------------------\n'
    print '\n'
    for item in classification: print item

    # Print the contents of the decision tree
    print '\n'
    print '------------------------\n'
    print '--   Decision Tree    --\n'
    print '------------------------\n'
    print '\n'
    print_tree(dtree, '')


if __name__ == '__main__':
    main()
