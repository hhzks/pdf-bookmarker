const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

// XMLHttpRequest instead of fetch: fetch has no upload-progress events.
export function createJob(file, { llmMode, model, apiKey }, onProgress) {
  const form = new FormData();
  form.append("file", file);
  form.append("llm_mode", llmMode);
  if (model) form.append("model", model);
  if (apiKey) form.append("api_key", apiKey);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/api/jobs`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status === 202) resolve(xhr.response.job_id);
      else reject(new Error(xhr.response?.detail ?? `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Network error during upload."));
    xhr.send(form);
  });
}

export async function getJob(jobId) {
  const res = await fetch(`${BASE}/api/jobs/${jobId}`);
  if (res.status === 404) {
    throw new Error("This job has expired — please upload the file again.");
  }
  if (!res.ok) throw new Error(`Status check failed (${res.status}).`);
  return res.json();
}

export function downloadUrl(jobId) {
  return `${BASE}/api/jobs/${jobId}/download`;
}
