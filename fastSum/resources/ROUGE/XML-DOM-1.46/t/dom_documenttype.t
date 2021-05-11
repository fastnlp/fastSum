BEGIN {print "1..1\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
$loaded = 1;
my $xml = new XML::DOM::Document;
$xml->setDoctype($xml->createDocumentType('Sample', 'Sample.dtd'));
print "not " unless $xml->toString eq qq{<!DOCTYPE Sample SYSTEM "Sample.dtd">\n};
print "ok 1\n";
