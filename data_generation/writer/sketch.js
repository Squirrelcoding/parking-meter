const WIDTH = 1000;
const HEIGHT = 1000;
const NUM_CARS = 10;


function setup() {
  createCanvas(WIDTH, HEIGHT);
  angleMode("radians");
}

function draw() {
  data = {
    cars: [
      { "dimensions": [446.7525446512403, 181.47669061058784], "corner_radius": 26.325571969531563, "color": "black", "front_windshield": { "color": [27, 44, 107], "windshield_width": 167.8098978130811, "windshield_length": 102.82113014789313, "hood_length": 67.54682616182586 }, "rear_windshield": { "color": [24, 38, 79], "windshield_width": 154.9241243424313, "windshield_length": 0.17546914771091462, "trunk_length": 34.44768718298192 }, "effects": { "shine": null, "roof_squares": [], "roof_lines": [], "transparency": 1.0, "clutter_pixels": [], "saturation": 0, "shadow": null } }]
  }

  console.log(data);
  for (const car of data.cars) {
    console.log("HERE!");
    x = 200
    y = 200

    // Draw the base
    fill(car.color);
    rect(x, y, car.dimensions[0], car.dimensions[1], car.corner_radius)

    // Draw the forward windshield
    fill(car.front_windshield.color);
    rect(x + car.front_windshield.hood_length,
      y + (car.dimensions[1] - car.front_windshield.windshield_width) / 2,
      car.front_windshield.windshield_length,
      car.front_windshield.windshield_width,
      car.corner_radius
    )

    // Draw the rear windshield
    console.log(car.rear_windshield.color);
    fill(car.rear_windshield.color);
    rect(x + car.dimensions[0] - (car.rear_windshield.trunk_length + car.rear_windshield.windshield_length),
      y + (car.dimensions[1] - car.rear_windshield.windshield_width) / 2,
      car.rear_windshield.windshield_length,
      car.rear_windshield.windshield_width,
      car.corner_radius
    )

    filter(BLUR);
  }

  noLoop();
}

// const colors = [
//   [175, 172, 153],
//   [186, 184, 173],
//   [178, 176, 166],
//   [215, 218, 214],
//   [181, 174, 161],
//   [212, 212, 205],
//   [191, 191, 186],
//   [188, 186, 177],
//   [74, 75, 69],
//   [184, 181, 166],
//   [176, 173, 161],
//   [186, 188, 183],
//   [91, 89, 63],
//   [0, 0, 0],
//   [207, 203, 189],
//   [192, 192, 184],
//   [197, 194, 183],
//   [191, 188, 175],
//   [169, 164, 147],
//   [207, 206, 202],
//   [177, 175, 162],
//   [182, 177, 165],
//   [54, 67, 66],
//   [186, 182, 169],
//   [185, 181, 168],
//   [173, 170, 157],
//   [72, 78, 74],
//   [213, 207, 193],
//   [189, 186, 174],
// ];

// function getSize() {
//   const color = colors[Math.floor(Math.random() * colors.length)];
//   return color;
// }

// function getHoodSize() {
//   const color = colors[Math.floor(Math.random() * colors.length)];
//   return color;
// }

// function getRooftopWindowSize() {
//   const color = colors[Math.floor(Math.random() * colors.length)];
//   return color;
// }

// function getRandomLines() {
//   const color = colors[Math.floor(Math.random() * colors.length)];
//   return color;
// }

// function getColor() {
//   const color = colors[Math.floor(Math.random() * colors.length)];
//   return color;
// }
