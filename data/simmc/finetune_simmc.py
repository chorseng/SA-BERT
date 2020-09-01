# coding=utf-8
import collections
import tokenization
import tensorflow as tf
from tqdm import tqdm
import json

tf.flags.DEFINE_string("train_response_file", "./fashion_train_dials_retrieval_candidates.json", 
                       "path to response file")
tf.flags.DEFINE_string("valid_response_file", "./fashion_dev_dials_retrieval_candidates.json", 
                       "path to response file")
tf.flags.DEFINE_string("test_response_file", "./fashion_devtest_dials_retrieval_candidates.json", 
                       "path to response file")
tf.flags.DEFINE_string("train_file", "./fashion_train_dials.json", 
	                   "path to train file")
tf.flags.DEFINE_string("valid_file", "./fashion_dev_dials.json", 
	                   "path to valid file")
tf.flags.DEFINE_string("test_file", "./fashion_devtest_dials.json", 
                       "path to test file")
tf.flags.DEFINE_string("vocab_file", "../../uncased_L-12_H-768_A-12/vocab.txt", 
                       "path to vocab file")
tf.flags.DEFINE_integer("max_seq_length", 512, 
	                    "max sequence length of concatenated context and response")
tf.flags.DEFINE_bool("do_lower_case", True,
                     "whether to lower case the input text")
                     
def print_configuration_op(FLAGS):
    print('My Configurations:')
    for name, value in FLAGS.__flags.items():
        value=value.value
        if type(value) == float:
            print(' %s:\t %f'%(name, value))
        elif type(value) == int:
            print(' %s:\t %d'%(name, value))
        elif type(value) == str:
            print(' %s:\t %s'%(name, value))
        elif type(value) == bool:
            print(' %s:\t %s'%(name, value))
        else:
            print('%s:\t %s' % (name, value))
    print('End of configuration')
        
    
def load_responses(fname):
    responses={}
    with open(fname, 'r') as f:
	response_dict = json.load(f)
        for response_idx, response_text in enumerate(response_dict['system_transcript_pool']):
            responses[response_idx] = response_text
    return responses

def get_candidates(fname):
    with open(fname, 'r') as f:
        json_dict = json.load(f)
						     
    return json_dict
						     
						     
def extract_dialog(fname):
    print("Extacting dialogs from {} ...".format(fname))
    dialogs = []						    
    with open(fname, 'r') as f:
        dial_json = json.load(f)
    for dialog in dial_json['dialogue_data']:
        dialog_dict = {}
	dialog_dict['dialog_idx'] = dialog['dialogue_idx']
	dialog_dict['dialog'] = []
	utterances = ""
	for turn in dialog['dialogue']:
	    utterances = utterances + turn['transcript']
	    dialog_dict['dialog'].append(utterances)
	    utterances = utterances + ' __EOS__ ' + turn['system_transcript'] + ' __EOS__ '
        dialogs.append(dialog_dict)
						     
    return dialogs
						     
def load_dataset(dialogs, candidates, responses, suffix):

    processed_fname = "processed_" + suffix
    dataset_size = 0
    print("Generating the file of {} ...".format(processed_fname))
    
    with open(processed_fname, 'w') as fw:						     
						     
    	response_idx = 0
	us_id = 0					    
    	for dialog in dialogs:
	    dialog_idx = dialog['dialog_idx']
	    if dialog_idx != candidates['retrieval_candidates'][response_idx]['dialogue_idx']:
	        print("DIALOGS DON'T MATCH!")
	    else:
	        turn_idx = 0
                for turn_context in dialog['dialog']:
	    	    context = turn_context
		
		    pos_ids = candidates['retrieval_candidates'][response_idx]['retrieval_candidates'][turn_idx]['retrieval_candidates'][0]
		    r_utter = responses[pos_ids]
		    dataset_size += 1
		    fw.write("\t".join([str(us_id), context.encode('ascii', 'ignore').decode('ascii'), str(pos_ids), r_utter.encode('ascii', 'ignore').decode('ascii'), "follow"]))
		    fw.write('\n')
			
		    for neg_ids in candidates['retrieval_candidates'][response_idx]['retrieval_candidates'][turn_idx]['retrieval_candidates'][1:]:
		        r_utter = responses[neg_ids]
			dataset_size += 1
			#print(str(us_id) + " " + context + " " + str(neg_ids) + " " + r_utter + " " + "unfollow")
		        fw.write("\t".join([str(us_id), context.encode('ascii', 'ignore').decode('ascii'), str(neg_ids), r_utter.encode('ascii', 'ignore').decode('ascii'), "unfollow"]))
			fw.write('\n')
			
		    us_id += 1			     
		turn_idx += 1				
	    response_idx += 1	     
						     
    print("Num_lines = " + str(len(open(processed_fname).readlines())))
    print("{} dataset_size: {}".format(processed_fname, dataset_size))            
    return processed_fname


class InputExample(object):
    def __init__(self, guid,ques_ids, text_a, ans_ids, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.ques_ids = ques_ids
        self.ans_ids = ans_ids
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, ques_ids, ans_ids, input_sents, input_mask, segment_ids, switch_ids, label_id):
        self.ques_ids = ques_ids
        self.ans_ids = ans_ids
        self.input_sents = input_sents
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.switch_ids=switch_ids
        self.label_id = label_id

def read_processed_file(input_file):
    """Counts number of lines in input file, concatenates all the information in each line into a list, and appends to another list (lines). """
    lines = []
    num_lines = sum(1 for line in open(input_file, 'r'))
    with open(input_file, 'r') as f:
        for line in tqdm(f, total=num_lines):
            concat = []
            temp = line.rstrip().split('\t')
            concat.append(temp[0]) # contxt id
            concat.append(temp[1]) # contxt
            concat.append(temp[2]) # response id
            concat.append(temp[3]) # response
            concat.append(temp[4]) # label
            lines.append(concat)
    return lines

def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, str(i)) # e.g. "train-1"
        ques_ids = line[0]
	#print(ques_ids)
        text_a = tokenization.convert_to_unicode(line[1])
	#print(text_a)
        ans_ids = line[2]
	#print(ans_ids)
        text_b = tokenization.convert_to_unicode(line[3])
	#print(text_b)
        label = tokenization.convert_to_unicode(line[-1])
	#print(label, line[-1])
        examples.append(InputExample(guid=guid, ques_ids=ques_ids, text_a=text_a, ans_ids=ans_ids, text_b=text_b, label=label))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}  # label
    for (i, label) in enumerate(label_list):  # ['0', '1'] so "unfollow" = 0 and "follow" = 1
        label_map[label] = i

    features = []  # feature
    for (ex_index, example) in enumerate(examples):
        ques_ids = int(example.ques_ids)
        ans_ids = int(example.ans_ids)

        # tokens_a = tokenizer.tokenize(example.text_a)  # text_a tokenize
        text_a_utters = example.text_a.split(" __EOS__ ")
        tokens_a = []
        text_a_switch = []
        for text_a_utter_idx, text_a_utter in enumerate(text_a_utters):
            if text_a_utter_idx%2 == 0:
                text_a_switch_flag = 0 # user utterance is tagged 0
            else:
                text_a_switch_flag = 1 # systerm utterance is tagged 1
            text_a_utter_token = tokenizer.tokenize(text_a_utter + " __EOS__") #tokenizes the entire sentence utterance, I think it will split into individual words
            tokens_a.extend(text_a_utter_token)  # adds to tokens_a
            text_a_switch.extend([text_a_switch_flag]*len(text_a_utter_token)) 
        assert len(tokens_a) == len(text_a_switch)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)  # text_b tokenize

        if tokens_b:  # if has b
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)  # truncate
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because  # (?)
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        switch_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        switch_ids.append(0)
        for token_idx, token in enumerate(tokens_a):
            tokens.append(token)
            segment_ids.append(0)
            switch_ids.append(text_a_switch[token_idx])
        tokens.append("[SEP]")
        segment_ids.append(0)
        switch_ids.append(0)

        if tokens_b:
            for token_idx, token in enumerate(tokens_b):
                tokens.append(token)
                segment_ids.append(1)
                switch_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            switch_ids.append(1)

        input_sents = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_sents)  # mask

        # Zero-pad up to the sequence length.
        while len(input_sents) < max_seq_length:
            input_sents.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            switch_ids.append(0)

        assert len(input_sents) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(switch_ids) == max_seq_length
	
	#print(example.ques_ids)
	#print(example.label)
        label_id = label_map[example.label]

        if ex_index%2000 == 0:
            print('convert_{}_examples_to_features'.format(ex_index))

        features.append(
            InputFeatures(  # object
                ques_ids=ques_ids,
                ans_ids = ans_ids,
                input_sents=input_sents,
                input_mask=input_mask,
                segment_ids=segment_ids,
                switch_ids=switch_ids,
                label_id=label_id))

    return features


def write_instance_to_example_files(instances, output_files):
    writers = []

    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        features = collections.OrderedDict()
        features["ques_ids"] = create_int_feature([instance.ques_ids])
        features["ans_ids"] = create_int_feature([instance.ans_ids])
        features["input_sents"] = create_int_feature(instance.input_sents)
        features["input_mask"] = create_int_feature(instance.input_mask)
        features["segment_ids"] = create_int_feature(instance.segment_ids)
        features["switch_ids"] = create_int_feature(instance.switch_ids)
        features["label_ids"] = create_float_feature([instance.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

    print("write_{}_instance_to_example_files".format(total_written))

    for feature_name in features.keys():
        feature = features[feature_name]
        values = []
    if feature.int64_list.value:
        values = feature.int64_list.value
    elif feature.float_list.value:
        values = feature.float_list.value
    tf.logging.info(
        "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()


def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature

def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature

    
if __name__ == "__main__":

    FLAGS = tf.flags.FLAGS
    print_configuration_op(FLAGS)
    
    print("Processing candidate responses ...")
    train_responses = load_responses(FLAGS.train_response_file)
    valid_responses = load_responses(FLAGS.valid_response_file)
    test_responses = load_responses(FLAGS.test_response_file)
    print("Processing candidate responses done!")
    
    print("Extracting dialogs ...")
    train_dials = extract_dialog(FLAGS.train_file)
    valid_dials = extract_dialog(FLAGS.valid_file)
    test_dials = extract_dialog(FLAGS.test_file)
    print("Dialog extraction done!")
	
	
    print("Extracting candidates ... ")					     
    train_candidates = get_candidates(FLAGS.train_response_file)
    valid_candidates = get_candidates(FLAGS.valid_response_file)
    test_candidates = get_candidates(FLAGS.test_response_file)
    print("Extracting candidates done!")
    
    print("Loading datasets ...")
    train_filename = load_dataset(train_dials, train_candidates, train_responses, 'train')
    valid_filename = load_dataset(valid_dials, valid_candidates, valid_responses, 'valid')
    test_filename  = load_dataset(test_dials, test_candidates, test_responses, 'test')
    print("Datasets loaded!")

    filenames = [train_filename, valid_filename, test_filename]
    filetypes = ["train", "valid", "test"]
    files = zip(filenames, filetypes)

    label_list = ["unfollow", "follow"]
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    for (filename, filetype) in files:
	print("Now processing " + filename + " ...")
        examples = create_examples(read_processed_file(filename), filetype)
        features = convert_examples_to_features(examples, label_list, FLAGS.max_seq_length, tokenizer)
        new_filename = filename[:-4] + ".tfrecord"
        write_instance_to_example_files(features, [new_filename])
        print('Convert {} to {} done'.format(filename, new_filename))

    print("Sub-process(es) done.")
