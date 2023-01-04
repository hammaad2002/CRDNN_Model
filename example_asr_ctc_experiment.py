#!/usr/bin/env/python3
"""

This minimal example trains a CTC-based speech recognizer on a tiny dataset.
The encoder is based on a combination of convolutional, recurrent, and
feed-forward networks (CRDNN) that predict phonemes.  A greedy search is used on
top of the output probabilities.
Given the tiny dataset, the expected behavior is to overfit the training dataset
(with a validation performance that stays high).

"""

import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

class CTCBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        wavs, lens = batch.sig 
        '''
        This len function returns a row vector and the number of elements in it 
        #is equal to the batch size, or the number of voice samples in a batch. 
        #The value at each index is equal to the length of the recording of 
        #this sample divided by the length of the recording of the longest 
        #sample. For example, if there are 3 audio recordings in the batch and 
        #their lengths are 1, 2, and 3 seconds respectively, then len will be 
        #[0.33, 0.66, 1] in this way.
        '''
        #print("\nThis is lens:",lens,'\n')
        #print("\nThis is wav:",wavs,'\n')
        '''
        likewise wav is also just the audio in vector format
        '''
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x = self.modules.model(feats)
        x = self.modules.lin(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the CTC loss."
        predictions, lens = predictions
        #print("This is prediction:",predictions)
        #print("\nThis is lens1:",lens,'\n')
        phns, phn_lens = batch.phn_encoded
        #print("\nThis is phn lens:",lens,'\n')
        decoded_phonemes = batch.phn_decoded
        #print("This is batch encoded phonemes:",batch.phn_encoded)
        #print("This is simple encoded phonemes:",phns)
        #print("This is batch decoded phonemes:",decoded_phonemes)
        loss = self.hparams.compute_cost(predictions, phns, lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            seq = sb.decoders.ctc_greedy_decode(
                predictions, lens, blank_id=self.hparams.blank_index
            )
            self.per_metrics.append(batch.id, seq, phns, target_len=phn_lens)
        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        
        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["PER"] = self.per_metrics.summarize("error_rate")
        if stage == sb.Stage.VALID and epoch is not None:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch+1},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current+1},
                test_stats=stage_stats,
            )
            with open(self.hparams.per_file, "w") as f:
              self.per_metrics.write_stats(f)


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path= hparams["json_train"] ,
        replacements={"data_root": data_folder} ,
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path= hparams["json_train"] ,
        replacements={"data_root": data_folder} ,
    )
    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        #print("wav signal:",wav) #/content/CRDNN_Model/AudioSamplesASR/spk1_snt1.wav
        sig = sb.dataio.dataio.read_audio(wav)
        #print("Signal:",sig)   # <=== tensor([-9.1553e-05, -9.1553e-05 and len
        return sig   #  <=== len , sign_vec = sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")  
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded","phn_decoded")
    def text_pipeline(phn):
        #print("This is phns",phn) ##s ah n vcl d ey
        phn_list = phn.strip().split()
        #print("This is phns list",phn_list)  #['s','ah','n','vcl']
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        #print("Encoded list",phn_encoded) #[1,2,3,4]
        phn_decoded = label_encoder.decode_torch(phn_encoded)
        #print("Decoded list",phn_decoded) #['s','ah','n','vcl']
        yield phn_encoded
        yield phn_decoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data
    label_encoder.insert_blank(index=hparams["blank_index"])
    label_encoder.update_from_didataset(train_data, output_key="phn_list")
    label_encoder.update_from_didataset(valid_data, output_key="phn_list")

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded", "phn_decoded"])

    return train_data, valid_data


def main(device="cpu"):
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"



    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
      
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        save_env_desc = True,
    )

    data_folder = hparams["data_folder"]
    data_folder = (experiment_dir / data_folder).resolve()

    # Dataset creation
    train_data, valid_data = data_prep(data_folder, hparams)

    # Trainer initialization
    ctc_brain = CTCBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        run_opts={"device": device},
    )

    # Training/validation loop
    ctc_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    ctc_brain.evaluate(valid_data)

    # Check if model overfits for integration test
    # Assert ctc_brain.train_stats < 1.0 # ignoring overfitting case right now

if __name__ == "__main__":
    main()

def test_error(device):
    main(device)