// Dependency-free CDP transport for a dedicated acceptance-test browser.
import readline from 'node:readline';
import {createHash} from 'node:crypto';
const [endpoint, origin] = process.argv.slice(2);
if (!endpoint || !origin) throw new Error('web_match_cdp.mjs requires page WebSocket URL and game origin');
const socket = new WebSocket(endpoint);
let nextId = 1;
const pending = new Map(), requests = new Map(), receipts = [];
const output = value => process.stdout.write(JSON.stringify(value) + '\n');
function call(method, params = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params}));
  });
}
socket.addEventListener('message', async event => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const task = pending.get(message.id);
    if (task) {
      pending.delete(message.id);
      message.error ? task.reject(new Error(JSON.stringify(message.error))) : task.resolve(message.result || {});
    }
    return;
  }
  const p = message.params || {};
  if (message.method === 'Network.requestWillBeSent' && p.request.url.startsWith(origin + '/api/')) {
    requests.set(p.requestId, {request_id:p.requestId, url:p.request.url,
      method:p.request.method, post_data:p.request.postData || null,
      request_timestamp:p.timestamp, initiator_type:p.initiator?.type});
  } else if (message.method === 'Network.responseReceived' && requests.has(p.requestId)) {
    Object.assign(requests.get(p.requestId), {status:p.response.status, response_timestamp:p.timestamp});
  } else if (message.method === 'Network.loadingFinished' && requests.has(p.requestId)) {
    const request = requests.get(p.requestId);
    requests.delete(p.requestId);
    try {
      const result = await call('Network.getResponseBody', {requestId:p.requestId});
      const body = result.base64Encoded ? Buffer.from(result.body, 'base64').toString('utf8') : result.body;
      receipts.push({...request, body:JSON.parse(body), body_bytes:Buffer.byteLength(body, 'utf8'),
        body_sha256:createHash('sha256').update(body, 'utf8').digest('hex')});
    } catch (error) {
      receipts.push({...request, error:String(error)});
    }
  } else if (message.method === 'Network.loadingFailed' && requests.has(p.requestId)) {
    receipts.push({...requests.get(p.requestId), error:p.errorText});
    requests.delete(p.requestId);
  }
});
socket.addEventListener('error', event => output({fatal:String(event.message || 'CDP socket failed')}));
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, {once:true});
  socket.addEventListener('error', reject, {once:true});
});
await call('Page.enable');
await call('Runtime.enable');
await call('Network.enable', {maxTotalBufferSize:128*1024*1024, maxResourceBufferSize:64*1024*1024});
output({ready:true});
for await (const line of readline.createInterface({input:process.stdin, crlfDelay:Infinity})) {
  if (!line.trim()) continue;
  let request;
  try {
    request = JSON.parse(line);
    let result;
    if (request.method === '__receipts') result = {receipts:receipts.splice(0)};
    else result = await call(request.method, request.params || {});
    output({id:request.id, result});
  } catch (error) {
    output({id:request?.id, error:String(error)});
  }
}
socket.close();
