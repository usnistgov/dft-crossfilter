/* jshint node: true */
/* global describe, it, beforeEach */
'use strict';

var eco    = require('../');
var should = require('should');
var gutil  = require('gulp-util');
require('mocha');

describe('gulp-eco-template', function() {
  var fakeFile;

  function getFakeFile(fileContent){
    return new gutil.File({
      path: './test/fixture/file.eco',
      cwd: './test/',
      base: './test/fixture/',
      contents: new Buffer(fileContent || '')
    });
  }

  beforeEach(function(){
    fakeFile = getFakeFile('<%= "Hello #{@input}!!" %>');
  });

  describe('eco', function() {
    it('file should pass through', function(done) {
      var file_count = 0;
      var stream = eco({input: 'eco'});
      stream.on('data', function(newFile){
        should.exist(newFile);
        should.exist(newFile.path);
        should.exist(newFile.relative);
        should.exist(newFile.contents);
        newFile.path.should.equal('./test/fixture/file.eco');
        newFile.relative.should.equal('file.eco');
        ++file_count;
      });

      stream.once('end', function () {
        file_count.should.equal(1);
        done();
      });

      stream.write(fakeFile);
      stream.end();
    });


    it('should render the eco template', function(done) {
      var stream = eco({input: 'eco'});
      stream.on('data', function (newFile) {
        should.exist(newFile.contents);
        newFile.contents.toString().should.equal('Hello eco!!');
      });
      stream.once('end', done);

      stream.write(fakeFile);
      stream.end();
    });


    // it('should format the eco', function(done) {
    //   var stream = eco('And then <%= foo %> said : ', { foo : 'you' } );
    //   //var stream = eco('And then ${foo} said : ', { foo : 'you' } );
    //   stream.on('data', function (newFile) {
    //     should.exist(newFile.contents);
    //     newFile.contents.toString().should.equal('And then you said : Hello world');
    //   });
    //   stream.once('end', done);

    //   stream.write(fakeFile);
    //   stream.end();
    // });


    // it('should format the eco (ES6 delimiters)', function(done) {
    //   var stream = eco('And then ${foo} said : ', { foo : 'you' } );
    //   stream.on('data', function (newFile) {
    //     should.exist(newFile.contents);
    //     newFile.contents.toString().should.equal('And then you said : Hello world');
    //   });
    //   stream.once('end', done);

    //   stream.write(fakeFile);
    //   stream.end();
    // });


    // it('should access to the current file', function(done) {
    //   var stream = eco([
    //     '<%= file.relative %>',
    //     '<%= file.path %>',
    //     ''].join('\n'));
    //   stream.on('data', function (newFile) {
    //     should.exist(newFile.contents);
    //     newFile.contents.toString().should.equal('file.js\n./test/fixture/file.js\nHello world');
    //   });
    //   stream.once('end', done);

    //   stream.write(fakeFile);
    //   stream.end();
    // });

  });

});
