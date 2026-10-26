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

const individual = new SceneTracker({confirmFrames:1,exitFrames:2,exitMilliseconds:100});
const a={...wallet,box:{x:.1,y:.2,w:.1,h:.1}},b={...wallet,box:{x:.7,y:.2,w:.1,h:.1}};
const first=individual.update({detections:[a,b]},0);
const reordered=individual.update({detections:[b,a]},10);
assert.equal(first.detections[0].id,reordered.detections.find(d=>d.box.x<.5).id);
individual.update({detections:[a]},100);
const lost=individual.update({detections:[a]},200);
assert.equal(lost.detections.length,1);
assert.equal(lost.events.filter(e=>e.type==='exit').length,1);
console.log('INDIVIDUAL_TRACKING_OK');
