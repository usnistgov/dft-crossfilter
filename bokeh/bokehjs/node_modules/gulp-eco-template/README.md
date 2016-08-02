gulp-eco-template
=================

Gulp plugin to render [eco](https://github.com/sstephenson/eco) embeded
CoffeeScript templates.

Install
-------

    npm install --save-dep gulp-eco-template

Usage
------

It accepts a data object for passing helpers and variables.

    var gulp = require('gulp');
    var template = require('gulp-eco-template');

    gulp.task('eco', function(){
      gulp.src('./src/templates/*.html')
          .pipe(template({author: 'Macario'}))
          .pipe(gulp.dest('./build'));
    });

    gulp.task('default', ['eco']); 
  
See [eco](https://github.com/sstephenson/eco) for eco templates syntax.
