import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
import torchaudio
import sys

class CTCBrain(sb.Brain):
    def compute_forward(self, batch, stage, n = 0):
        if n == 1:
          wavs = batch
          wavs = wavs.to("cuda")
          lens = torch.tensor([1.])
          feats = self.modules.compute_features(wavs) 
          feats = self.modules.mean_var_norm(feats, lens)
          x = self.modules.model(feats)
          x = self.modules.lin(x)
          predictions = {"ctc_softmax": self.hparams.softmax(x)}
          predictions["seq"] = self.hparams.decoder(
                  predictions["ctc_softmax"], lens, blank_id=0)
          return predictions, lens       
        else:
          batch = batch.to(self.device)
          wavs, lens = batch.sig
          feats = self.modules.compute_features(wavs)
          feats = self.modules.mean_var_norm(feats, lens)
          x = self.modules.model(feats)
          x = self.modules.lin(x)
          predictions = {"ctc_softmax": self.hparams.softmax(x)}
          predictions["seq"] = self.hparams.decoder(
                   predictions["ctc_softmax"], lens, blank_id=self.hparams.blank_index)
          return predictions, lens

    def compute_objectives(self, predictions, batch, stage):
        predictions, lens = predictions
        phns, phn_lens = batch.phn_encoded
        decoded_phonemes = batch.phn_decoded
        label = batch.label_encoder
        label_encoder = label[0]
        loss = self.hparams.compute_cost(predictions["ctc_softmax"], phns, lens, phn_lens)
        if stage != sb.Stage.TRAIN:
            output = predictions["seq"]
            seq = output
            output1 = torch.tensor(output) 
            output = label_encoder.decode_torch(output1)
            self.per_metrics.append(batch.id, seq, phns, target_len=phn_lens,ind2lab = lambda x: label_encoder.decode_torch(torch.tensor(x)))
        return loss

    def transcribe_dataset(
            self,
            dataset, 
            min_key, 
            label
          ):
        data_waveform, rate_of_sample = torchaudio.load(dataset)
        samples = data_waveform
        self.on_evaluate_start(min_key=min_key)
        self.modules.eval() 
        with torch.no_grad():
                out = self.compute_forward(samples, stage=sb.Stage.TEST, n = 1) 
                p_seq, wav_lens = out
        output = p_seq["seq"]
        output = torch.tensor(output)
        output = label.decode_torch(output)
        return output

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["PER"] = self.per_metrics.summarize("error_rate")
        if stage == sb.Stage.VALID and epoch is not None:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch+1},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": stage_stats["PER"]}, min_keys=["PER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current+1},
                test_stats=stage_stats,
            )
            with open(self.hparams.per_file, "w") as f:
              self.per_metrics.write_stats(f)

def data_prep(data_folder, hparams, n = 0):
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path= hparams["json_train"] ,
        replacements={"data_root": data_folder} ,
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path= hparams["json_dev"] ,
        replacements={"data_root": data_folder} ,
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path= hparams["json_test"] ,
        replacements={"data_root": data_folder} ,
    )
    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    @sb.utils.data_pipeline.takes("phn")  
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded","phn_decoded","label_encoder")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        mapped_phonemes = {
            "iy": "iy",
            "ix": "ix",
            "ih": "ix",
            "eh": "eh",
            "ae": "ae",
            "ax": "ax",
            "ah": "ax",
            "ax-h": "ax",
            "uw": "uw",
            "ux": "uw",
            "uh": "uh",
            "ao": "ao",
            "aa": "ao",
            "ey": "ey",
            "ay": "ay",
            "oy": "oy",
            "aw": "aw",
            "ow": "ow",
            "er": "er",
            "axr": "er",
            "l": "l",
            "el": "l",
            "r": "r",
            "w": "w",
            "y": "y",
            "m": "m",
            "em": "m",
            "n": "n",
            "en": "n",
            "nx": "n",
            "ng": "ng",
            "eng": "ng",
            "v": "v",
            "f": "f",
            "dh": "dh",
            "th": "th",
            "z": "z",
            "s": "s",
            "zh": "zh",
            "sh": "zh",
            "jh": "jh",
            "ch": "ch",
            "b": "b",
            "p": "p",
            "d": "d",
            "dx": "dx",
            "t": "t",
            "g": "g",
            "k": "k",
            "hh": "hh",
            "hv": "hh",
            "bcl": "h#",
            "pcl": "h#",
            "dcl": "h#",
            "tcl": "h#",
            "gcl": "h#",
            "kcl": "h#",
            "q": "h#",
            "epi": "h#",
            "pau": "h#",
            "h#": "h#"
            }
        def map_phonemes(original_phonemes):
          mapped_phonemes_list = []
          for phoneme in original_phonemes:
            mapped_phoneme = mapped_phonemes.get(phoneme, None)
            if mapped_phoneme:
              mapped_phonemes_list.append(mapped_phoneme)
          return mapped_phonemes_list
        phn_list = map_phonemes(phn_list)
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        phn_decoded = label_encoder.decode_torch(phn_encoded)
        yield phn_encoded
        yield phn_decoded
        yield label_encoder
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    label_encoder.insert_blank(index=hparams["blank_index"])
    label_encoder.update_from_didataset(train_data, output_key="phn_list")
    label_encoder.update_from_didataset(valid_data, output_key="phn_list")
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded", "phn_decoded","label_encoder"])
    if n == 1:
      return label_encoder
    else:
      return train_data, valid_data, test_data

def main(device="cuda"):
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        save_env_desc = True,
    )
    data_folder = hparams["data_folder"]
    data_folder = (experiment_dir / data_folder).resolve()
    train_data, valid_data, test_data = data_prep(data_folder, hparams, n = 0)
    label_encoder = data_prep(data_folder, hparams, n = 1)
    ctc_brain = CTCBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"],
    )    
    #c1
    ctc_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    ctc_brain.evaluate(test_data,
    min_key="PER",
    )
    #c2
    '''
    transcripts = ctc_brain.transcribe_dataset(
        dataset= overrides,
        min_key="PER",
        label = label_encoder 
    )
    print(transcripts)
    '''
if __name__ == "__main__":
    main()
def test_error(device):
    main(device)