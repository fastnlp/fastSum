use ExtUtils::MakeMaker;

sub MY::libscan
{
    package MY;

    my ($self, $file) = @_;

    # Don't install these PM files (or Emacs or other backups: *~ *.bak)
    # Also don't install XML/Parser.pod and XML/Parser/Expat.pod because I copied
    # those from the XML::Parser distribution.
    return undef if $file =~ /(XML.Parser\.pod|Expat\.pod|CmpDOM|CheckAncestors|~$|\.bak$)/;

    return $self->SUPER::libscan ($file);
}

# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    NAME	=> 'XML::DOM',
    VERSION_FROM => 'lib/XML/DOM.pm',
                   # XML::Parser 2.28 and above work, but make test
                   # doesn't pass because of different ways
                   # errors are reported
    PREREQ_PM => { 'XML::Parser' => '2.30',
		   # LWP::UserAgent is used when parsing XML from URLs
		   # It's part of libwww-perl, and you don't strictly need it
		   # (some test cases may fail)
		   'LWP::UserAgent' => '0',
		   # XML::Parser::PerlSAX is part of libxml-perl.
		   # It's used by some test cases in t/chk_batch.t and you 
		   # don't strictly need it. Version 0.05 causes errors in the
		   # test cases in t/chk_batch.t.
		   'XML::Parser::PerlSAX' => '0.07',
		   'XML::RegExp' => 0,
		   },
    dist        => {'COMPRESS'  => 'gzip', 'SUFFIX'    => '.gz'},
);
