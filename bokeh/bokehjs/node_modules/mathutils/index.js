var MathUtils = module.exports = {
	isOdd: function(num){
		return num & 1 === 1;
	},
	isEven: function(num){
		return num & 1 === 0;
	},

	powermod: function powermod(num, exp, mod){
		if(exp === 1) return num % mod;
		if(MathUtils.isOdd(exp)){
			return (num * powermod(num, exp-1, mod)) % mod;
		}
		return Math.pow(powermod(num, exp/2, mod), 2) % mod;
	},

	isPrime: function(num){
		return MathUtils.fastIsPrime(num) && MathUtils.slowIsPrime(num);
	},
	slowIsPrime: function(num){
		if(MathUtils.isEven(num)) return false;
		for(var i = 3, max = Math.sqrt(num); i < max; i += 2){
			if(num % i === 0) return false;
		}
		return true;
	},
	fastIsPrime: function(num){
		return MathUtils.powermod(3, num-1, num) === 1;
	},

	randomPrime: function(len){
		var num = Math.floor(Math.pow(10, len || 3) * Math.random());
		if(MathUtils.isEven(num)) num++;
		while(!MathUtils.isPrime(num)) num += 2;
		return num;
	},

	gcd: function gcd(a, b){
		if(b === 0) return a;
		return gcd(b, a % b);
	},
	egcd: function eea(a, b){
		if(b === 0) return [a, 1, 0];
		var tmp = eea(b, a % b);
		var ss = tmp[1],
			ts = tmp[2];
		return [tmp[0], ts, ss - Math.floor(a/b) * ts];
	},

	modularInverse: function(a, b){
		var arr = MathUtils.egcd(a, b);
		//if(arr[1] * a + arr[2] * b !== arr[0]) throw Error("Wrong EGCD: " + sum);
		return arr[1];
	}
};