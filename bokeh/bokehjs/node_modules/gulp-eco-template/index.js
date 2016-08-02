/* jshint node: true */
'use strict';

var es    = require('event-stream');
var eco   = require('eco');
var gutil = require('gulp-util');

var ecoTemplatePlugin = function(locals, data) {
  locals = locals || {};

  return es.map(function(file, cb){
    file.contents =
      new Buffer( eco.render(file.contents.toString(), locals) );
    cb(null, file);
  });
};

module.exports = ecoTemplatePlugin;
