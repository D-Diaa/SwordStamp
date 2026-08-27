We use standard attack methods implemented by MarkLLM. Therefore to conduct attack experiments, just put ``attack_utils.py`` and ``attack_pmark.py`` in the root dir of MarkLLM and pass the log dir when running ``attack_pmark.py``.

The attacker we use is defined in ``attack_utils.py``. For paraphraser Parrot, install the lib:
```
pip install git+https://github.com/PrithivirajDamodaran/Parrot_Paraphraser.git
``` 