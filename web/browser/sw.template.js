/** Build substitutes a content-derived version and the complete local asset list. */
const VERSION=__VERSION__;
const FILES=__FILES__;
const SCOPE=self.registration.scope;
const PREFIX='must5-browser-'+encodeURIComponent(SCOPE)+'-';
const CACHE=PREFIX+VERSION;
self.addEventListener('install',event=>event.waitUntil((async()=>{
  const cache=await caches.open(CACHE);
  try{await cache.addAll(FILES);}catch(error){await caches.delete(CACHE);throw error;}
})()));
self.addEventListener('activate',event=>event.waitUntil((async()=>{
  for(const key of await caches.keys()){
    if(key.startsWith(PREFIX)&&key!==CACHE)await caches.delete(key);
    // Migrate an old unscoped cache only when every entry belongs to this app.
    if(/^must5-browser-[0-9a-f]{20}$/.test(key)){
      const entries=await (await caches.open(key)).keys();
      if(entries.length&&entries.every(request=>request.url.startsWith(SCOPE)))await caches.delete(key);
    }
  }
  await self.clients.claim();
})()));
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET'||!event.request.url.startsWith(SCOPE))return;
  event.respondWith((async()=>{
    const cache=await caches.open(CACHE);let response=await cache.match(event.request,{ignoreSearch:true});
    if(!response&&event.request.mode==='navigate')response=await cache.match('./index.html');
    return response||fetch(event.request);
  })());
});
self.addEventListener('message',event=>{
  if(event.data?.type!=='cache-status')return;
  event.waitUntil((async()=>{const cache=await caches.open(CACHE);
    const ready=(await Promise.all(FILES.map(file=>cache.match(file)))).every(Boolean);
    event.ports[0]?.postMessage({ready,version:VERSION});})());
});
