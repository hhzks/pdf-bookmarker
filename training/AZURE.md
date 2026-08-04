# Azure GPU box for the 8k/3B retrain

Azure-specific instance of the generic runbook in [REMOTE.md](REMOTE.md): same
goal (retrain with `--max-seq-len 8192` on `Qwen/Qwen2.5-3B-Instruct`), with
the provisioning, dataset transfer, and teardown scripted under
`training/azure/`.

Target box: **Standard_NC4as_T4_v3** — 4 vCPU, 28 GB RAM, 1× Tesla T4 (16 GB),
**$0.558/hr** pay-as-you-go in `swedencentral`. At 16 GB this is the "likely
fits" tier in REMOTE.md's VRAM table, not the comfortable one; if 8192 OOMs,
drop to `MAX_SEQ_LEN=6144` before falling back to the 1.5B base.

## 0. Status: blocked on the student subscription

The subscription in use is `Azure for Students`
(`quotaId=AzureForStudents_2018-01-01`, on a UCL tenant). The resource group
and storage are provisioned and verified; **the GPU VM cannot be created on this
subscription at all**, for the two reasons below. Steps 1–6 are ready for a
subscription that has GPU quota.

**Blocker 1 — region placement policy.** Of 17 regions probed, `swedencentral`
is the only one that accepts a deployment. The other 16 — eastus, eastus2,
uksouth, westeurope, northeurope, francecentral, centralus, northcentralus,
southcentralus, westus, westus2, westus3, canadacentral, japaneast,
southeastasia, australiaeast — reject *any* resource type with:

```
(RequestDisallowedByAzure) ... This policy maintains a set of best available
regions where your subscription can deploy resources.
```

Note `az account list-locations` still lists all 63 physical regions — it does
not reflect this policy, so the only way to test a region is to try creating
something in it. That is why `LOC` defaults to `swedencentral` in
`provision.sh` and why the resource group lives there.

**Blocker 2 — GPU quota is zero.** In swedencentral (and every other region
checked) every modern GPU family has limit 0:

| Quota (swedencentral) | Limit |
|---|---|
| Total Regional vCPUs | 6 |
| Standard NCASv3_T4 Family vCPUs | **0** |
| Standard NCADS_A100_v4 Family vCPUs | 0 |
| Standard NCADSA10v4 / NVADSA10v5 Family | 0 |
| Spot / low-priority vCPUs | not offered |

Azure ML's separate compute-cluster quota is the same story: only the retired
NC/NV v1 (K80/M60) families show non-zero limits, and those SKUs are no longer
deployable anywhere.

Confusingly, the SKUs themselves are *offered*: in swedencentral
`Standard_NC4as_T4_v3` and `Standard_NC24ads_A100_v4` both come back
**unrestricted** from `az vm list-skus`. Only the quota number is 0.

**The quota increase was requested and refused at the API level.** Asking for 4
vCPU on the T4 family (and, as a cross-check, 24 on the A100 family) both
return:

```
(ResourceNotAvailableForOffer) Request failed.
```

That is not a request queued for human review — the Azure for Students offer
simply excludes GPU quota, so there is no number to raise and no ticket to
wait on. For the record, the request was:

```bash
az extension add --name quota   # and: az provider register -n Microsoft.Quota
az quota update --resource-name "Standard NCASv3_T4 Family" --resource-type dedicated \
    --scope "/subscriptions/$(az account show --query id -o tsv)/providers/Microsoft.Compute/locations/swedencentral" \
    --limit-object value=4
```

(Note the quota resource name contains literal spaces — `standardNCASv3_T4Family`
is rejected as an invalid resource name.)

**So this subscription cannot train the model, and no amount of setup here will
change that.** Nothing in `training/azure/` is student-offer specific though:
point it at a subscription with GPU quota — pay-as-you-go, or a UCL
departmental/research one — with

```bash
az account set --subscription "<name or id>"
LOC=uksouth ./training/azure/provision.sh
```

A non-student subscription almost certainly has no region placement policy, so
set `LOC` to something near you. Re-probe first: create a throwaway vnet in the
target region and confirm it is not `RequestDisallowedByAzure`.

On a pay-as-you-go sub the T4 is still the right size — $0.558/hr, so a 2-epoch
8k run costs a few dollars. Keep `spendingLimit` in mind only for the student
sub: it hard-stops at the credit balance rather than billing over.

Providers `Microsoft.Compute`, `Microsoft.Network`, `Microsoft.Storage`, and
`Microsoft.MachineLearningServices` are registered already.

## 1. Provision

```bash
./training/azure/provision.sh
```

Already done and verified: the resource group `pdfbm-train` and the storage
account (`pdfbm<hash>`, derived from the subscription id) with its `artifacts`
container, both in swedencentral. Re-running skips them and goes straight to the
VM — which is the step that needs the quota from step 0.

Idempotent — re-run it any time to bring the box back or re-narrow the SSH
rule. It creates the resource group (`pdfbm-train`), a Standard_LRS storage
account plus `artifacts` container for moving files, and the VM: Ubuntu 22.04
gen2, 128 GB StandardSSD (the 30 GB default cannot hold torch plus a 6 GB base
model plus checkpoints), the `NvidiaGpuDriverLinux` extension for CUDA, and a
system-assigned identity.

SSH is opened only to this machine's current public IP. Re-run the script if
your IP moves.

Override anything via env: `RG`, `LOC`, `VM`, `SIZE`, `OS_DISK_GB`, `STORAGE`.

## 2. Why the identity

The VM gets `Storage Blob Data Contributor` on the storage account and
`Virtual Machine Contributor` **on itself**. That second one lets
`run_training.sh` deallocate the box as its last action, so a finished run
stops billing without anyone watching it. A guest-side `shutdown` would not:
Azure keeps charging for a VM that is stopped but not deallocated.

## 3. Push the dataset

The GPU box needs no PDFs — only `dataset/{train,val}.jsonl`. Note this Mac has
no `corpus/`, `records.jsonl`, or `dataset/` in the working tree, so the bundle
has to come from the machine that harvested the corpus (the 12 GB Windows box).
Build it there, then from either machine:

```bash
zip -r dataset-bundle.zip dataset/
./training/azure/push_dataset.sh dataset-bundle.zip
```

## 4. Train

```bash
ssh azureuser@<ip>
tmux new -s train                     # a dropped SSH kills the run
curl -fsSL -o run_training.sh \
  https://raw.githubusercontent.com/hhzks/pdf-bookmarker/master/training/azure/run_training.sh
bash run_training.sh
```

It logs in with the managed identity, clones the repo, builds a venv, installs
`training/requirements.txt`, pulls the dataset bundle, trains, uploads
`outline-lora-8k.zip`, and deallocates the VM. Pass `DEALLOCATE=0` to keep the
box up (it keeps billing), or `MAX_SEQ_LEN=6144` / `BASE_MODEL=...` to adjust.

Sanity-check the first stderr line — `train: N (dropped M over-long)` — the
drop count should be near zero at 8192, versus 413 of 873 at 4096.

Note for the T4: `finetune.py` now picks the bitsandbytes compute dtype from
the card's capability (fp16 on pre-Ampere, bf16 on Ampere and newer). It used
to hardcode bf16, which Turing does not support.

## 5. Fetch the adapter

```bash
./training/azure/fetch_adapter.sh
```

Then continue locally with `predict.py` → `evaluate.py` → `export_gguf.py`
(README.md steps 4–7). Pass `--base-model Qwen/Qwen2.5-3B-Instruct` to
`predict.py` and `export_gguf.py` — the default is still the 1.5B.

## 6. Stop paying

```bash
./training/azure/teardown.sh stop      # between runs: deallocate, keep the disk
./training/azure/teardown.sh delete    # done for good: delete the group
```

`run_training.sh` already deallocates on a clean finish. Use `stop` after a run
that crashed or that you interrupted, and check with:

```bash
az vm list -d -g pdfbm-train -o tsv --query "[].[name,powerState]"
```
