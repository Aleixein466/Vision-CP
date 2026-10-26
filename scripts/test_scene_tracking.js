const assert = require('assert');
const { SceneTracker } = require('../web/scene-tracker.js');

const tracker = new SceneTracker({ confirmFrames: 2, exitFrames: 3, exitMilliseconds: 1500 });
const wallet = { class: 'wallet', label: 'billetera', confidence: .9, box: { x: .4, y: .4, w: .2, h: .2 } };
let result = tracker.update({ detections: [wallet] }, 0);
assert.equal(result.events.length, 0);
result = tracker.update({ detections: [wallet] }, 100);
assert.deepEqual(result.events.map((event) => event.type), ['enter']);
for (let time = 200; time <= 1200; time += 200) {
  result = tracker.update({ detections: [wallet] }, time);
  assert.equal(result.events.length, 0, 'No debe anunciar de nuevo mientras siga visible');
}
tracker.update({ detections: [] }, 1300);
tracker.update({ detections: [] }, 1500);
result = tracker.update({ detections: [] }, 2800);
assert.deepEqual(result.events.map((event) => event.type), ['exit']);
tracker.update({ detections: [wallet] }, 3000);
result = tracker.update({ detections: [wallet] }, 3200);
assert.deepEqual(result.events.map((event) => event.type), ['enter']);
console.log('SCENE_TRACKING_OK');
