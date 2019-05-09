from utils.PreProcess import linear2mu,  mu2linear
from ruamel.yaml import YAML
import tensorflow as tf
import numpy as np
import dcase_util
import os



class dict_to_object(object):
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

    def __str__(self):
        return "\n".join(["{} : {}".format(k,v) for k,v in self.__dict__.items()])


# https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/
# https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def count_examples(record_list):
    return sum([1 for record in record_list for _ in tf.python_io.tf_record_iterator(record)])
  
def data_to_tfrecords(db, file_list, files_dir, data_dir, meta_dir_file, 
    event_label, segment_length, segment_overlap):
    meta_file = data_meta_parse("{}{}.yaml".format(meta_dir_file, event_label))
    scene_dict = dict()
    for file_name in file_list: 
        file_scene = meta_file[file_name].bg_classname.replace('/', '_')
        if file_scene in scene_dict:
            scene_dict[file_scene].append(file_name)
        else:
            scene_dict[file_scene] = [file_name]

    for file_scene in scene_dict:
    # for file_scene in scene_dict:
        scene_dir = os.path.join(data_dir, file_scene)
         # Get directory for specific file and write it for each scene if it doesn't exist. 
        # Make subdirectories for scenes. 
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)  
        # Negative examples in a separate file to avoid mistakes. 
        pos_writer = tf.python_io.TFRecordWriter(
            os.path.join(scene_dir, "{}_{}_pos.tfrecord".format(event_label, file_scene))
            )
        neg_writer = tf.python_io.TFRecordWriter(
            os.path.join(scene_dir, "{}_{}_neg.tfrecord".format(event_label, file_scene))
            )
        

        for file_name in scene_dict[file_scene]:
            file_path = os.path.join(files_dir, file_name)
            # Replace slash as one scene contains slash which messes with directories ("cafe/restaurant"). 
            assert file_scene == meta_file[file_name].bg_classname.replace('/', '_')

            # dcase_util.containers.metadata.MetaDataContainer
            items = db.file_meta(filename=file_path)
            # MetaDataContainer
            audio = dcase_util.containers.AudioContainer().load( 
                filename=file_path, mono=True)
            # Float, needs to be converted to an int. 
            duration_in_secs = int(audio.duration_sec)
            sampling_rate = audio.fs

            # segment_meta is a MetaDataContainer
            # Parameter exists to skip segments: 
            # https://dcase-repo.github.io/dcase_util/tutorial_audio.html
            # Non-overlapping frames
            # data, segment_meta = audio.segments(segment_length_seconds=segment_length)
            data = audio.frames(
                frame_length=segment_length, 
                hop_length=segment_overlap).T
            # Used to be event_roll encoder, pass single label for binary or all for one-hot. 
            event_roll = dcase_util.data.EventRollEncoder(
                label_list= event_label,
                time_resolution=(1/sampling_rate)
                ).encode(
                metadata_container=items, 
                length_seconds=duration_in_secs
                )
            # Sum along the row axis as classes are mutually exclusive. 
            event_roll = np.sum(event_roll.data, axis=0)
            # Calculate the amount of "extra" examples there will be. 
            over = len(event_roll) % segment_length
            # Make sure that the the event roll - the amount of overlap is how data was framed. 
            assert data.shape[0] * data.shape[1] == len(event_roll)-over
            # Remove samples 'over' at the end (librosa frame does the same thing for the data)
            event_roll = event_roll[:len(event_roll)-over]
            # Split into the same number of batches as the segments. 
            labels = np.array(np.split(event_roll, data.shape[0]))
            # Get most common label for this segment. 
            labels = np.amax(labels, axis=1)
            # Reduce to binary vector instead of one hot, classes only occur one at a time. 
            # Outputs
            data_quantised = [linear2mu(d) for d in data]

            # Inputs
            # numpy.float64
            # data_quantised_centred = [d /128-1 for d in data_quantised]
            for segment in range(len(data_quantised)):  
                event_present =  'true' if labels[segment] == 1 else 'false'
                print(labels[segment])
                features={
                  # 'label': _bytes_feature(label.tostring()),
                  'label': _int64_feature(labels[segment].astype(np.int64)),
                  'audio_inputs': _bytes_feature(data_quantised[segment].tostring()),
                  # 'audio_outputs': _bytes_feature(data_quantised[segment].tostring()),
                  'scene': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_scene))),
                  'source_file': _bytes_feature(tf.compat.as_bytes(
                        '{}_{}'.format(meta_file[file_name].mixture_audio_filename.split('.')[0], segment)
                    )
                  )
                  }
                
                example = tf.train.Example(features=tf.train.Features(feature=features))
                if labels[segment] == 1:
                    pos_writer.write(example.SerializeToString())
                else:
                    neg_writer.write(example.SerializeToString())
        pos_writer.close()
        neg_writer.close()
        print("Scene {} written to {}".format(file_scene, scene_dir))



def write_data(segment_length, segment_overlap, data_dir, binaries_dir):

    if not os.path.exists(binaries_dir):
        os.makedirs(binaries_dir)
    else:
        print('Binaries are already written to {}'.format(binaries_dir))
        return 0

    db = dcase_util.datasets.TUTRareSoundEvents_2017_DevelopmentSet(
    data_path=data_dir
    )


    eval_db = dcase_util.datasets.TUTRareSoundEvents_2017_EvaluationSet(
    data_path=data_dir
    )

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        # Download and prepare datasets: https://dcase-repo.github.io/dcase_util/tutorial_datasets.html
        db.initialize()
        eval_db.initialize()

    with open(os.path.join(data_dir,"TUT-rare-sound-events-2017-development",
        "TUT_Rare_sound_events_mixture_synthesizer/affected_files.txt"), "r") as f:
        erroneous_files = f.read().splitlines()

    devtrain_dir = os.path.join(data_dir,"TUT-rare-sound-events-2017-development/data",
        "mixture_data/devtrain/20b255387a2d0cddc0a3dff5014875e7")
    devtest_dir = os.path.join(data_dir,"TUT-rare-sound-events-2017-development/data",
        "mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7")
    evaltest_dir = os.path.join(data_dir,"TUT-rare-sound-events-2017-evaluation/data",
        "mixture_data/evaltest/bbb81504db15a03680a0044474633b67")

    train_meta_dir = os.path.join(devtrain_dir,"meta")
    val_meta_dir  = os.path.join(devtest_dir,"meta")
    test_meta_dir = os.path.join(evaltest_dir,"meta")


    train_audio_dir = os.path.join(devtrain_dir,"audio")
    val_audio_dir  = os.path.join(devtest_dir,"audio")
    test_audio_dir = os.path.join(evaltest_dir,"audio")

    train_meta_dir_file = os.path.join(train_meta_dir,"mixture_recipes_devtrain_")
    val_meta_dir_file = os.path.join(val_meta_dir,"mixture_recipes_devtest_")
    test_meta_dir_file = os.path.join(test_meta_dir,"mixture_recipes_evaltest_")

    dir_names = ('train', 'val', 'test')
    event_labels = db.event_labels()
    
    # Make directories for each event label so they can be tested individually. 
    for event_label in event_labels:
        for name in dir_names:
            event_dir = os.path.join(binaries_dir, name, event_label)
            if not os.path.exists(event_dir):
                os.makedirs(event_dir)

        event_train_dir = os.path.join(binaries_dir,'train', event_label)
        event_val_dir = os.path.join(binaries_dir,'val', event_label)
        event_test_dir = os.path.join(binaries_dir,'test', event_label)

        train_event_list_data_dir = os.path.join(train_meta_dir, "event_list_devtrain_"+event_label+".csv")
        val_event_list_data_dir = os.path.join(val_meta_dir, "event_list_devtest_"+event_label+".csv")
        test_event_list_data_dir= os.path.join(test_meta_dir, "event_list_evaltest_"+event_label+".csv")

        train_list = get_file_list(train_event_list_data_dir)
        val_list = get_file_list(val_event_list_data_dir)
        eval_list = get_file_list(test_event_list_data_dir)
        
        # In the development dataset, there were erroneous files that we must
        # make sure are gone. 
        assert len(set(erroneous_files).intersection(set(train_list))) == 0
        assert len(set(erroneous_files).intersection(set(val_list))) == 0

        test_overlaps()


        data_to_tfrecords(
            db, 
            train_list, 
            train_audio_dir,
            event_train_dir, 
            train_meta_dir_file, 
            event_label, 
            segment_length, 
            segment_overlap
            )

        data_to_tfrecords(
            db, 
            val_list, 
            val_audio_dir,
            event_val_dir, 
            val_meta_dir_file, 
            event_label, 
            segment_length, 
            segment_overlap
            )
       
        data_to_tfrecords(
            eval_db, 
            eval_list, 
            test_audio_dir,
            event_test_dir, 
            test_meta_dir_file, 
            event_label, 
            segment_length, 
            segment_overlap
            )


def get_file_list(csv_file):
    with open(csv_file) as meta_f:
        event_list = [row.split()[0] for row in meta_f]
    return event_list

def data_meta_parse(yaml_file):
    with open(yaml_file) as f:
        yaml = YAML(typ='safe')
        yaml_map = yaml.load(f)
        set_dict = dict()
        for file_dict in yaml_map:
            set_dict[file_dict['mixture_audio_filename']] = dict_to_object(file_dict)
        return set_dict


def test_overlaps():
    data_dir = 'data'
    devtrain_dir = os.path.join(data_dir,"TUT-rare-sound-events-2017-development/data",
        "mixture_data/devtrain/20b255387a2d0cddc0a3dff5014875e7/")
    devtest_dir = os.path.join(data_dir,"TUT-rare-sound-events-2017-development/data",
        "mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7")
    evaltest_dir = os.path.join(data_dir,"TUT-rare-sound-events-2017-evaluation/data",
        "mixture_data/evaltest/bbb81504db15a03680a0044474633b67")

    train_meta_dir = os.path.join(devtrain_dir,"meta")
    val_meta_dir  = os.path.join(devtest_dir,"meta")
    test_meta_dir = os.path.join(evaltest_dir,"meta")

    train_meta_dir_file = os.path.join(train_meta_dir,"mixture_recipes_devtrain_")
    val_meta_dir_file = os.path.join(val_meta_dir,"mixture_recipes_devtest_")
    test_meta_dir_file = os.path.join(test_meta_dir,"mixture_recipes_evaltest_")
   
    train_audio_dir = os.path.join(devtrain_dir,"audio")
    val_audio_dir  = os.path.join(devtest_dir,"audio")
    test_audio_dir = os.path.join(evaltest_dir,"audio")
 
    # Get unique files for each event class from 
    b_train_files = get_file_list(os.path.join(train_meta_dir, "event_list_devtrain_babycry.csv"))
    gs_train_files =get_file_list(os.path.join(train_meta_dir, "event_list_devtrain_gunshot.csv"))
    gb_train_files =get_file_list(os.path.join(train_meta_dir, "event_list_devtrain_glassbreak.csv"))

    
    # Create metadata dictionary with the file name as the dict key for each event class set. 
    b_train_meta = data_meta_parse(train_meta_dir_file+"babycry.yaml")
    gs_train_meta = data_meta_parse(train_meta_dir_file+"gunshot.yaml")
    gb_train_meta = data_meta_parse(train_meta_dir_file+"glassbreak.yaml")

    # Get source audio files for each file in each set. 
    b_train_scene_source = set([b_train_meta[file.split('/')[-1]].bg_path for file in b_train_files])
    gs_train_scene_source = set([gs_train_meta[file.split('/')[-1]].bg_path for file in gs_train_files])
    gb_train_scene_source = set([gb_train_meta[file.split('/')[-1]].bg_path for file in gb_train_files])

    # Get scene class for each set. 
    b_train_scenes = set([b_train_meta[file.split('/')[-1]].bg_classname for file in b_train_files])
    gs_train_scenes = set([gs_train_meta[file.split('/')[-1]].bg_classname for file in gs_train_files])
    gb_train_scenes = set([gb_train_meta[file.split('/')[-1]].bg_classname for file in gb_train_files])

    # Check that the same files are not across different sets. 
    assert len(set(b_train_files).intersection(gs_train_files)) == 0
    assert len(set(b_train_files).intersection(gb_train_files)) == 0
    assert len(set(gb_train_files).intersection(gs_train_files)) == 0

    # Gather all scene classes. 
    train_scenes = b_train_scenes | gs_train_scenes | gb_train_scenes

    b_val_files = get_file_list(os.path.join(val_meta_dir, "event_list_devtest_babycry.csv"))
    gs_val_files =get_file_list(os.path.join(val_meta_dir, "event_list_devtest_gunshot.csv"))
    gb_val_files =get_file_list(os.path.join(val_meta_dir, "event_list_devtest_glassbreak.csv"))

    b_val_meta = data_meta_parse(val_meta_dir_file+"babycry.yaml")
    gs_val_meta = data_meta_parse(val_meta_dir_file+"gunshot.yaml")
    gb_val_meta = data_meta_parse(val_meta_dir_file+"glassbreak.yaml")

    b_val_scene_source = set([b_val_meta[file.split('/')[-1]].bg_path for file in b_val_files])
    gs_val_scene_source = set([gs_val_meta[file.split('/')[-1]].bg_path for file in gs_val_files])
    gb_val_scene_source = set([gb_val_meta[file.split('/')[-1]].bg_path for file in gb_val_files])

    b_val_scenes= set([b_val_meta[file.split('/')[-1]].bg_classname for file in b_val_files])
    gs_val_scenes = set([gs_val_meta[file.split('/')[-1]].bg_classname for file in gs_val_files])
    gb_val_scenes = set([gb_val_meta[file.split('/')[-1]].bg_classname for file in gb_val_files])

    val_scenes = b_val_scenes | gs_val_scenes | gb_val_scenes

    assert len(set(b_val_files).intersection(set(gs_val_files))) == 0
    assert len(set(b_val_files).intersection(set(gb_val_files))) == 0
    assert len(set(gs_val_files).intersection(set(gb_val_files))) == 0
    
    # Check that the source audio files for scenes do not overlap in the test and validation set for each event class. 
    assert len(b_val_scene_source.intersection(b_train_scene_source)) == 0
    assert len(gs_val_scene_source.intersection(gs_train_scene_source)) == 0
    assert len(gb_val_scene_source.intersection(gb_train_scene_source)) == 0

    # Check that the source audio files for scenes do not overlap in the test and validation set for all event classes together.     
    train_scene_source = b_train_scene_source | gs_train_scene_source | gb_train_scene_source
    val_scene_source = b_val_scene_source | gs_val_scene_source | gb_val_scene_source
    assert len(train_scene_source.intersection(val_scene_source)) == 0

    b_test_files = get_file_list(os.path.join(test_meta_dir, "event_list_evaltest_babycry.csv"))
    gs_test_files =get_file_list(os.path.join(test_meta_dir, "event_list_evaltest_gunshot.csv"))
    gb_test_files =get_file_list(os.path.join(test_meta_dir, "event_list_evaltest_glassbreak.csv"))
    
    b_test_meta = data_meta_parse(test_meta_dir_file+"babycry.yaml")
    gs_test_meta = data_meta_parse(test_meta_dir_file+"gunshot.yaml")
    gb_test_meta = data_meta_parse(test_meta_dir_file+"glassbreak.yaml")

    b_test_scene_source = set([b_test_meta[file.split('/')[-1]].bg_path for file in b_test_files])
    gs_test_scene_source = set([gs_test_meta[file.split('/')[-1]].bg_path for file in gs_test_files])
    gb_test_scene_source = set([gb_test_meta[file.split('/')[-1]].bg_path for file in gb_test_files])

    b_test_scenes = set([b_test_meta[file.split('/')[-1]].bg_classname  for file in b_test_files])
    gs_test_scenes = set([gs_test_meta[file.split('/')[-1]].bg_classname  for file in gs_test_files])
    gb_test_scenes = set([gb_test_meta[file.split('/')[-1]].bg_classname  for file in gb_test_files])

    test_scenes = b_test_scenes | gs_test_scenes | gb_test_scenes
    
    assert len(set(b_test_files).intersection(gs_test_files)) == 0
    assert len(set(b_test_files).intersection(gb_test_files)) == 0
    assert len(set(gb_test_files).intersection(gs_test_files)) == 0

    assert len(b_train_scene_source.intersection(b_test_scene_source)) == 0
    assert len(gs_train_scene_source.intersection(gs_test_scene_source)) == 0
    assert len(gb_train_scene_source.intersection(gb_test_scene_source)) == 0

    assert len(b_val_scene_source.intersection(b_test_scene_source)) == 0
    assert len(gs_val_scene_source.intersection(gs_test_scene_source)) == 0
    assert len(gb_val_scene_source.intersection(gb_test_scene_source)) == 0

    test_scene_source = b_test_scene_source | gs_test_scene_source | gb_test_scene_source
    assert len(train_scene_source.intersection(test_scene_source)) == 0
    assert len(val_scene_source.intersection(test_scene_source)) == 0

    # Check that there are scenes in common amoungst all sets. 
    assert train_scenes == val_scenes == test_scenes
    assert len(train_scenes | val_scenes | test_scenes) == 15

def _parse_function(file):
    features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'audio_inputs': tf.FixedLenFeature([], tf.string),
        'scene': tf.FixedLenFeature([], tf.string),
        'source_file': tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_single_example(file, features=features)
    
    label = tf.cast(parsed_features['label'], tf.int64), 
    audio_inputs = tf.decode_raw(parsed_features['audio_inputs'], tf.int64),
    audio_targets = tf.cast(audio_inputs, dtype=tf.float32)/128 - 1 
    scene = tf.decode_raw(parsed_features['scene'], tf.uint8), 
    source_file = tf.decode_raw(parsed_features['source_file'], tf.uint8),
    
    return (
        label , 
        audio_inputs,
        audio_targets,
        scene, 
        source_file
        )
