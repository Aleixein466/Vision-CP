(function attachSceneTracker(scope) {
  function iou(a, b) {
    const area = Math.max(0, Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x)) * Math.max(0, Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));
    return area / Math.max(1e-9, a.w*a.h+b.w*b.h-area);
  }
  class SceneTracker {
    constructor(options = {}) {
      this.confirmFrames=options.confirmFrames || 2;
      this.exitFrames=options.exitFrames || 3;
      this.exitMilliseconds=options.exitMilliseconds || 1500;
      this.tracks=new Map(); this.nextId=1;
    }
    clear() { this.tracks.clear(); this.nextId=1; }
    update(data, now=Date.now()) {
      const detections=data.detections || [], used=new Set(), matched=new Set(), events=[];
      const pairs=[];
      for (const [id,t] of this.tracks) detections.forEach((d,index)=>{
        if (d.class===t.item.class) {
          const score=iou(d.box,t.item.box);
          if(score>.15) pairs.push({id,index,score});
        }
      });
      pairs.sort((a,b)=>b.score-a.score);
      for(const pair of pairs) {
        if(used.has(pair.index)||matched.has(pair.id)) continue;
        used.add(pair.index); matched.add(pair.id);
        const t=this.tracks.get(pair.id), d=detections[pair.index];
        const box=Object.fromEntries(['x','y','w','h'].map(k=>[k,.65*d.box[k]+.35*t.item.box[k]]));
        t.item={...d,box,id:`track-${pair.id}`,track_id:pair.id};
        t.seen++; t.missing=0; t.lastSeen=now;
        if(!t.active && t.seen>=this.confirmFrames) { t.active=true; events.push({type:'enter',detections:[t.item]}); }
      }
      detections.forEach((d,index)=>{
        if(used.has(index)) return;
        const id=this.nextId++;
        const t={item:{...d,id:`track-${id}`,track_id:id},seen:1,missing:0,lastSeen:now,active:this.confirmFrames===1};
        this.tracks.set(id,t); matched.add(id);
        if(t.active) events.push({type:'enter',detections:[t.item]});
      });
      const visible=[];
      for(const [id,t] of this.tracks) {
        if(!matched.has(id)) {
          t.missing++; t.seen=0;
          if(t.missing>=this.exitFrames && now-t.lastSeen>=this.exitMilliseconds) {
            if(t.active) events.push({type:'exit',detections:[t.item]});
            this.tracks.delete(id); continue;
          }
        }
        if(t.active) visible.push({...t.item,stale:t.missing>0,uncertainty:t.missing?'temporarily_unobserved':t.item.uncertainty});
      }
      const counts={};
      visible.filter(d=>!d.stale).forEach(d=>{counts[d.label]=(counts[d.label]||0)+1;});
      return {...data,detections:visible,counts,events};
    }
  }
  scope.SceneTracker=SceneTracker;
  if(typeof module!=='undefined') module.exports={SceneTracker,iou};
})(typeof window!=='undefined'?window:globalThis);
