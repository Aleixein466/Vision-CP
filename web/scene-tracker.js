(function attachSceneTracker(globalScope) {
  class SceneTracker {
    constructor(options = {}) {
      this.confirmFrames = options.confirmFrames || 2;
      this.exitFrames = options.exitFrames || 3;
      this.exitMilliseconds = options.exitMilliseconds || 1500;
      this.tracks = new Map();
    }

    clear() {
      this.tracks.clear();
    }

    update(data, now = Date.now()) {
      const byClass = new Map();
      (data.detections || []).forEach((item) => {
        if (!byClass.has(item.class)) byClass.set(item.class, []);
        byClass.get(item.class).push(item);
      });
      const classes = new Set([...this.tracks.keys(), ...byClass.keys()]);
      const visible = [];
      const events = [];
      classes.forEach((className) => {
        const current = byClass.get(className) || [];
        const track = this.tracks.get(className) || { active: false, seen: 0, missing: 0, last: [], lastSeenAt: 0 };
        if (current.length) {
          track.seen += 1;
          track.missing = 0;
          track.last = current;
          track.lastSeenAt = now;
          if (!track.active && track.seen >= this.confirmFrames) {
            track.active = true;
            events.push({ type: 'enter', detections: current });
          }
          if (track.active) visible.push(...current);
        } else {
          track.seen = 0;
          if (track.active) {
            track.missing += 1;
            if (track.missing >= this.exitFrames && now - track.lastSeenAt >= this.exitMilliseconds) {
              track.active = false;
              events.push({ type: 'exit', detections: track.last });
              track.last = [];
            } else {
              visible.push(...track.last.map((item) => ({ ...item, stale: true })));
            }
          }
        }
        this.tracks.set(className, track);
      });
      const counts = visible.reduce((result, item) => {
        result[item.label] = (result[item.label] || 0) + 1;
        return result;
      }, {});
      return { ...data, detections: visible, counts, events };
    }
  }

  globalScope.SceneTracker = SceneTracker;
  if (typeof module !== 'undefined' && module.exports) module.exports = { SceneTracker };
})(typeof window !== 'undefined' ? window : globalThis);
