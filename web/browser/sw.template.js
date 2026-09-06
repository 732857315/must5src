/** Build substitutes a content-derived version and the complete local asset list. */
const VERSION=__VERSION__;
const FILES=__FILES__;
const CACHE='must5-browser-'+VERSION;
self.addEventListener('install',event=>event.waitUntil((async()=>{
  const cache=await caches.open(CACHE);
  try{await cache.addAll(FILES);}catch(error){await caches.delete(CACHE);throw error;}
})()));
self.addEventListener('activate',event=>event.waitUntil((async()=>{
  for(const key of await caches.keys())if(key.startsWith('must5-browser-')&&key!==CACHE)await caches.delete(key);
  await self.clients.claim();
})()));
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET'||new URL(event.request.url).origin!==self.location.origin)return;
  event.respondWith((async()=>{
    const cache=await caches.open(CACHE);let response=await cache.match(event.request,{ignoreSearch:true});
    if(!response&&event.request.mode==='navigate')response=await cache.match('./index.html');
    return response||fetch(event.request);
  })());
});
self.addEventListener('message',event=>{
  if(event.data?.type!=='cache-status')return;
  event.waitUntil((async()=>{const cache=await caches.open(CACHE);let ready=true;
    for(const file of FILES)if(!await cache.match(file)){ready=false;break;}
    event.ports[0]?.postMessage({ready,version:VERSION});})());
});
