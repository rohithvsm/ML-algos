import random
import sys
import os
import pprint
import collections
import math
import json

NUMERIC = 'numeric'


def parse_cmd_line_args():
    """Parse command-line arguments.

    neuralnet <data-set-file> n l e

    n -> number of folds for cross validation
    l -> learning rate
    e -> number of training epochs

    """

    if len(sys.argv) < 5:
        print >> sys.stderr, 'Argument error: Not enough arguments'
        sys.exit(1)

    data_set_file = sys.argv[1]
    num_folds = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    epochs = int(sys.argv[4])

    if not os.path.isfile(data_set_file):
        print >> sys.stderr, 'Argument error: Illegal arguments'
        sys.exit(1)

    # print data_set_file
    # print num_folds
    # print learning_rate
    # print epochs
    return data_set_file, num_folds, learning_rate, epochs


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
    # numeric_attrs = []
    # for attr in attributes_meta:
    #    if attributes_meta[attr] == NUMERIC:
    #        numeric_attrs.append(attr)
    with open(filename) as fin:
        for line in fin:
            if line.startswith('%'): continue
            if line.startswith('@'): continue
            features = line.strip().split(',')
            if features[-1] == attributes_meta[attributes[-1]][-1]:
                features[-1] = 1
            else:
                features[-1] = 0
            features = map(float, features)
            features[-1] = int(features[-1])
            data.append(features)
            # instance_dict = dict(zip(attributes,
            #                         [datum.strip() for datum in line.strip().
            #                         split(',')]))
            # for numeric_attr in numeric_attrs:
            #    val = instance_dict[numeric_attr]
            #    if '.' in val:
            #        instance_dict[numeric_attr] = float(val)
            #    else:
            #        instance_dict[numeric_attr] = int(val)
            # data.append(instance_dict)

    return data


class Perceptron:
    def __init__(self, learning_rate, attributes, attributes_meta, data_set):
        self.attributes_meta = attributes_meta
        self.attributes = attributes
        self.data_set = data_set
        self.bias = 0.1
        self.weights = [0.1 for attr in self.attributes[:-1]]  # weights
        self.learning_rate = learning_rate

        self.training_accuracies = collections.defaultdict(list)
        self.validation_accuracies = collections.defaultdict(list)

        self.roc_list = []
        self.num_neg = 0.0
        self.num_pos = 0.0

        self.output_lines = []

    def output(self, instance):
        """ perceptron output """
        net = self.bias
        for index, feature in enumerate(instance[:-1]):
            net += self.weights[index] * feature
        return self.sigmoid(net)

        # if y >= 0:
        #     return 1
        # else:
        #     return -1
        #
        #     def updateWeights(self, x, iterError):
        #         """
        #    updates the weights status, w at time t+1 is
        #        w(t+1) = w(t) + learningRate*(d-r)*x
        #    where d is desired output and r the perceptron response
        #    iterError is (d-r)
        #   """
        #         self.w[0] += self.learning_rate * iterError * x[0]
        #         self.w[1] += self.learning_rate * iterError * x[1]
        #

    def sigmoid(self, net):
        return 1 / (1 + math.e ** (-net))

    def calculate_accuracy(self, training, validation, epoch):
        num_training_instances = 0
        num_correct_training = 0
        for instance in training:
            num_training_instances += 1
            out = self.output(instance)
            if out > 0.5:
                out_class = 1
            else:
                out_class = 0
            if out_class == instance[-1]:
                num_correct_training += 1
        accuracy = float(num_correct_training) / float(num_training_instances)
        self.training_accuracies[epoch].append(accuracy)
        # print '%s: training' % epoch
        # print json.dumps(self.training_accuracies, indent=4)

        num_validation_instances = 0
        num_correct_validation = 0
        for instance in validation:
            num_validation_instances += 1
            out = self.output(instance)
            if out > 0.5:
                out_class = 1
            else:
                out_class = 0
            if out_class == instance[-1]:
                num_correct_validation += 1
        accuracy = float(num_correct_validation) / float(
            num_validation_instances)
        self.validation_accuracies[epoch].append(accuracy)
        # print '%s: validation' % epoch
        # print json.dumps(self.validation_accuracies, indent=4)

    def train(self, data, epochs):
        """
   trains all the vector in data.
   Every vector in data must have three elements,
   the third element (x[2]) must be the label (desired output)
  """

        fold_no = 0
        for training, validation in data:
            fold_no += 1  # print 1 to n rather than 0 to n-1
            for epoch in xrange(epochs):
                for instance in training:
                    # each sample
                    out = self.output(instance)
                    delta = self.learning_rate * (instance[-1] - out) * out * (
                        1 - out)
                    self.bias += delta
                    for idx, weight in enumerate(self.weights):
                        self.weights[idx] += delta * instance[idx]
                if epoch + 1 in (1, 10, 100, 1000):
                    self.calculate_accuracy(training, validation, epoch + 1)
            self.roc(validation)
            self.store_output_lines(validation, fold_no)
            self.bias = 0.1
            self.weights = [0.1 for attr in self.attributes[:-1]]

    def roc(self, validation):
        for instance in validation:
            output = self.output(instance)
            self.roc_list.append((output, instance[-1]))
            if instance[-1] == 0:
                self.num_neg += 1
            else:
                self.num_pos += 1

    def plot_roc_coordinates(self):
        # print '-' * 50
        self.roc_list.sort(reverse=True)
        # print pprint.pformat(self.roc_list)
        tp = 0
        fp = 0
        last_tp = 0
        for index in xrange(1, len(self.roc_list)):
            if (self.roc_list[index][0] != self.roc_list[index - 1][0]) and (
                        self.roc_list[index][1] == 0) and (tp > last_tp):
                fpr = fp / self.num_neg
                tpr = tp / self.num_pos
                print '%s %s' % (fpr, tpr)
                last_tp = tp
            if self.roc_list[index][1] == 1:
                tp += 1
            else:
                fp += 1
        fpr = fp / self.num_neg
        tpr = tp / self.num_pos
        print '%s %s' % (fpr, tpr)

    def store_output_lines(self, validation, fold_no):
        positive_class = self.attributes_meta[self.attributes[-1]][-1]
        negative_class = self.attributes_meta[self.attributes[-1]][0]
        for instance in validation:
            out = self.output(instance)
            if out > 0.5:
                predicted_class = positive_class
            else:
                predicted_class = negative_class
            output_line = (self.data_set.index(instance), fold_no,
                           predicted_class, (positive_class if instance[-1]
                                                               == 1 else
                                             negative_class), out)
            self.output_lines.append(output_line)

    def print_output_lines(self):
        self.output_lines.sort()
        for line in self.output_lines:
            print '%s,%s,%s,%s' % (line[1], line[2], line[3], line[4])

    def print_accuracies(self):
        print 'training accuracies:'
        for epoch in self.training_accuracies:
            print '%s: %s' % (epoch, sum(self.training_accuracies[epoch]) /
                              float(len(self.training_accuracies[epoch])))
        print 'validation accuracies:'
        for epoch in self.validation_accuracies:
            print '%s: %s' % (epoch, sum(self.validation_accuracies[epoch]) /
                              float(len(self.validation_accuracies[epoch])))


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def partition_data_set(data_set):
    for data_line in data_set:
        data_line.append(random.random())
    data_set.sort(key=lambda x: x[-1])
    chunks = collections.defaultdict(list)
    for index, data_line in enumerate(data_set):
        chunks[index % 10].append(data_line)
    return chunks


def k_fold_cross_validation(X, K, randomise=False):
    """
	Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
    if randomise: from random import shuffle; X = list(X); shuffle(X)
    chunks = []
    for k in xrange(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        # print training
        # print validation
        # print len(training) + len(validation)
        # sys.exit(0)
        chunks.append((training, validation))

    return chunks


# X = [i for i in xrange(97)]
# for training, validation in k_fold_cross_validation(X, K=7):
#     for x in X: assert (x in training) ^ (x in validation), x


def main():
    data_set_file, num_folds, learning_rate, epochs = parse_cmd_line_args()

    attributes, attributes_meta = get_attributes(data_set_file)
    data_set = get_data(data_set_file, attributes, attributes_meta)
    # print pprint.pformat(data_set)

    neg_chunks = k_fold_cross_validation([instance for instance in data_set if
                                          instance[-1] == 0], num_folds, True)
    pos_chunks = k_fold_cross_validation([instance for instance in data_set
                                          if instance[-1] == 1], num_folds,
                                         True)

    # print pprint.pformat(pos_chunks)
    # print pprint.pformat(len(neg_chunks))
    # print '-' * 60
    # print neg_chunks

    stratified_cv = []
    for neg_chunk, pos_chunk in zip(neg_chunks, pos_chunks):
        # print neg_chunk[1]
        # print pos_chunk[1]
        # print random.shuffle(neg_chunk[1])
        # break
        training_chunk = neg_chunk[0] + pos_chunk[0]
        random.shuffle(training_chunk)
        validation_chunk = neg_chunk[1] + pos_chunk[1]
        random.shuffle(validation_chunk)
        stratified_cv.append((training_chunk, validation_chunk))

    # print stratified_cv
    perceptron = Perceptron(learning_rate, attributes, attributes_meta,
                            data_set)
    perceptron.train(stratified_cv, epochs)  # training

    # perceptron.print_accuracies()
    # perceptron.plot_roc_coordinates()

    perceptron.print_output_lines()


if __name__ == '__main__':
    main()
