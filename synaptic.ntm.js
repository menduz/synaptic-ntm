//TODO:Fix memory allocations and releases. We are killing the garbage collector.

if(!'Synaptic' in this) throw "This module requires Synaptic";


var Utils = {};

Utils.createRandomWeights = function(size){
  var array = new Float64Array(size);
  Utils.fillRandomArrayUnsigned(array, size);
  return array;
}

Utils.fillRandomArrayUnsigned = function(array){
  if(array && 'length' in array && array.length)
    for(var i = 0; i < array.length; i++){
      array[i] = 0.0001 * Math.random();
    }
  return array;
}

Utils.fillRandomArraySigned = function(array){
  if(array && 'length' in array && array.length)
    for(var i = 0; i < array.length; i++){
      array[i] = 0.0002 * (Math.random() - 0.5);
    }
  return array;
}

Utils.softMaxArray = function(array, multiplyer){
  // for all i ∈ array
  // sum = ∑ array[n]^e
  // i = î^e / sum
  // where the result ∑ array[0..n] = 1

  if(!(array && 'length' in array && array.length)) return;

  multiplyer = multiplyer || 1;

  var sum = 0;

  // sum = ∑ array[n]^e
  for(var i = 0; i < array.length; i++){
    array[i] = Math.exp(multiplyer * array[i]);
    sum += array[i];
  }
  
  if(sum != 0){
    for(var i = 0; i < array.length; i++) array[i] /= sum;
  } else {
    var div = 1 / array.length;
    for(var i = 0; i < array.length; i++) array[i] = div;
  }

  return array;
}

Utils.normalizeArray = function(array){
  // normalize array to the R^(0..1) domain

  if(!(array && 'length' in array && array.length)) return;

  var max = null, min = null;

  // 
  for(var i = 0; i < array.length; i++){
    if(max === null || array[i] > max) max = array[i];
    if(min === null || array[i] < min) min = array[i];
  }

  var distance = max - min;

  if(distance == 0 && (min == 0 || max == 0)) return array;
  
  if(distance == 0 && max != 0) {
    for(var i = 0; i < array.length; i++)
      array[i] = 1;
    return array;
  };

  for(var i = 0; i < array.length; i++){
    array[i] = (array[i] - min) / distance;
  }

  return array;
}

Utils.dot = function(arrayA,arrayB){
  // f(a,b) = ∑ a[i] * b[i]
  var acum = 0;
  for(var i = 0; i < arrayA.length; i++){
    acum += arrayA[i] * arrayB[i];
  }
  return acum;
}

Utils.arrayMul = function(arrayA,arrayB){
  var ret = new Float32Array(arrayA.length);
  for(var i = 0; i < arrayA.length; i++){
    ret[i] = arrayA[i] * arrayB[i];
  }
  return ret;
}

Utils.arrayAdd = function(arrayA,arrayB){
  var ret = new Float32Array(arrayA.length);
  for(var i = 0; i < arrayA.length; i++){
    ret[i] = arrayA[i] + arrayB[i];
  }
  return ret;
}


// ROCK!: http://arxiv.org/pdf/1410.5401v1.pdf

function getSubArray(array, from, to){
	if('subarray' in array){
		return array.subarray(from, to);
	}
	return Array.prototype.slice.call(array, from, to);
}

function NTM(config){
	// config is an object with this fields:
	// config.controller {Network} a Synaptic.Network of any kind, this network should have at least (NTM.CONTROLLER_OUTPUT_SIZE + memWidth + shiftWidth) outputs
	// config.memWidth {Integer} how much data contains each memBlock ( M in paper )
	// config.memBlocks {Integer} how many memBlocks ( N in paper )
	// config.outputLength {Integer} how many outputs have the controller
	// config.numHeads {Integer} min 1. number of read/write heads

	if(!(this instanceof NTM)) throw "NTM is a class. You need use the 'new' operator"
	
	if(!config || !(config instanceof Object)) throw "args:config should be an object";

	this.config = config;

	this.controller = this.config.controller;

	this.config.numHeads = this.config.numHeads || 1;

	this.memory = [];

	for(var m = 0; m < config.memBlocks; m++)
		this.memory.push(
			//new Float64Array(config.memWidth)
			Utils.createRandomWeights(this.config.memWidth)
		);

	this.w = new Float64Array(this.config.memWidth); // focus on the memBlocks. w → ∑w[i] = 1

	// during the first stages we only have one a head
	this.heads = [];
	for(var head = 0; i < this.config.numHeads; i++)
		this.heads.push(new NTM.Head(this));

}

NTM.prototype.activate = function(inputs){
	var controllerOutputs = this.controller.activate(inputs);

	
	var B = controllerOutputs[0]; // key strength
	var g = controllerOutputs[1]; // interpolation gate
	var Y = controllerOutputs[2]; // depending on γ, the weighting is sharpened and used for memory access.
	var s = controllerOutputs[3]; // shifting vector
	var k = getSubArray(controllerOutputs, 4, 4 + this.config.memWidth); // key vector k

	var acumulatedHeadOutputs = [];

	for(var h = 0; h < this.heads.length; h++){
		acumulatedHeadOutputs.push(this.heads.getWeigthing(B, g, Y, k, s));
	}

	// compute erases over M
	///=paralelizable
	for(var h = 0; h < this.heads.length; h++){
		this.heads[h].doErase();
	}

	// compute adds over M
	///=paralelizable
	for(var h = 0; h < this.heads.length; h++){
		this.heads[h].doAdd();
	}

	return w;
}

NTM.CONTROLLER_OUTPUT_SIZE = 4; // [ß, g, γ, s] + w * outputs

/* Heads */

NTM.Head = function NTMHead(ntm){
	this.ntm = ntm;
	this.w = new Float64Array(ntm.config.memBlocks); // focus on the memBlocks. w → ∑w[i] = 1
	this.e = new Float64Array(ntm.config.memBlocks);
	this.a = new Float64Array(ntm.config.memBlocks);
	this.r = new Float64Array(ntm.config.memWidth);
}


NTM.Head.prototype.doAdd = function(){
	// 3.2 (4)
	for(var n = 0; n < this.ntm.memory.length; n++){
		var M = this.ntm.memory[n];
		for(var i = 0; i < M.length; i++){
			M[i] += this.a[i] * this.w[n];
		}
	}	
}

NTM.Head.prototype.doErase = function(){
	// 3.2 (3)
	for(var n = 0; n < this.ntm.memory.length; n++){
		var M = this.ntm.memory[n];
		for(var i = 0; i < M.length; i++){
			M[i] *= 1 - this.e[i] * this.w[n];
		}
	}	
}

NTM.Head.prototype.getWeigthing = function(
	B, // key strength
	g, // interpolation gate
	Y, // depending on γ, the weighting is sharpened and used for memory access.
	k, // key vector k
	s  // shift weighting, s, determines whether and by how much the weighting is rotated
){

	// 3.3.1 (5) & (6)
	// using similarity fn make an array[memBlocks] with index of similarity ∀ memBlock
	var similarityArray = this.getSimilarAdresses(k); // wc without normalization

 	// normalize similarityArray into wc using softmax
 	var wc = Utils.softMaxArray(similarityArray, B);

	// 3.3.2 (7)
	// interpolate wc into wg using interpolation gate (g)
	var wg = wg_interpolateWc(wc, g, this.w);

	// 3.3.2 (8)
	// shift interpolated wt (wg) using s
	var wn = wn_shift(wg, s);

	// 3.3.2 (9)
	// sharp using γ
	this.w = w_sharpWn(wn, Y);

	// 3.1 (2)
	// compute reading
	for(var i = 0; i < this.r.length; i++){
		var acum = 0, pointer = this.ntm.memory[i];
		for(var j = 0; j < this.w.length; j++){
			acum += pointer[j] * this.w[j];
		}
		this.r[i] = acum;
	}

	// compute erase gate
	//????
	// compute add gate
	//????


	// normalize erase gate and add gate to R^(0..1) domain
	normalizeGates(this.e);
	normalizeGates(this.a);

	return { w: this.w, erase: this.e, add: this.a, read: this.r };
}

NTM.Head.prototype.getSimilarAdresses = function(k){
	//checkpoint: 10th cigarret
	var addresses = new Float64Array(this.ntm.config.memWidth);

	for(var i = 0; i < this.ntm.memory.length; i++)
		addresess[i] = similarity(this.ntm.memory[i], k);

	return addresses;
}


/* HEAD HELPERS */


///=inline
function similarity(arrayA, arrayB){
	// http://en.wikipedia.org/wiki/Cosine_similarity
	// 3.3.1 (6)
	var dot = Utils.dot(arrayA, arrayB);

	var acumA = 0, acumB = 0;

	for(var i = 0; i < arrayA.length; i++){
		acumA += arrayA[i] * arrayA[i];
		acumB += arrayB[i] * arrayB[i];
	}

	return dot / (Math.sqrt(acumA) * Math.sqrt(acumB) + 0.00005);
}

///=inline
function normalizeGates(array){
	// normalize array to the R^(0..1) domain using sigmoid sqaushing fn.

	if(!(array && 'length' in array && array.length)) return;

	for(var i = 0; i < array.length; i++){
		array[i] = 1 / (1 + Math.exp(-array[i]));
	}

	return array;
}

///=inline
function wg_interpolateWc(wc, g, w){
	// 3.3.2 focus by location (7)
	// wg = interpolate wc with last w using g
	var gInverted = 1 - g;
	for(var i = 0; i < wc.length; i++)
		wc[i] = wc[i] * g + gInverted * w[i];
	return wc;
}

///=inline
function wn_shift(wg, shiftScalar){
	// w~ 3.3.2 (8)
	var shiftings = new Float64Array(wg.length);
	var wn = new Float64Array(wg.length);

	var intPart = shiftScalar | 0;
	var decimalPart = shiftScalar - intPart;

	shiftings[intPart % shiftings.length] = 1 - decimalPart;
	shiftings[(intPart + 1) % shiftings.length] = decimalPart;

	for(var i = 0; i < wn.length; i++){
		var acum = 0;
		for(var j = 0; j < wn.length; j++){
			acum += wg[j] * shiftings[(i - j) % shiftings.length];
		}
		wn[i] = acum;
	}

	return wn;
}

///=inline
function w_sharpWn(wn, Y){
	// 3.3.2 (9)

	var sum = 0;

	// ∀ a ∈ wn → a = a^Y
	// sum = ∑ a^Y 

	for(var i = 0; i < wn.length; i++){
		wn[i] = Math.pow(wn[i], Y);
		sum += wn[i];
	}

	// ∀ a ∈ wn → a = a^Y / sum
	if(sum != 0){
		for(var i = 0; i < wn.length; i++) wn[i] /= sum;
	} else {
		var div = 1 / wn.length;
		for(var i = 0; i < wn.length; i++) wn[i] = div;
	}

	return wn;
}


