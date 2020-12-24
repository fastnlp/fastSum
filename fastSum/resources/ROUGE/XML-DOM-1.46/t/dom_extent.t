BEGIN {print "1..1\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
$loaded = 1;
print "ok 1\n";

# this test is temporary disabled because
# i think i have found a bug in expat that
# calls the ExternEnt handler instead of Entity for
# external parameter entities
exit;

my $test = 1;
sub assert_ok
{
    my $ok = shift;
    print "not " unless $ok;
    ++$test;
    print "ok $test\n";
    $ok;
}

# <!ENTITY % globalInfo SYSTEM "t/dom_extent.ent">
my $xml =<<EOF;
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE book SYSTEM "t/dom_extent.dtd" [
 <!ENTITY % globalInfo SYSTEM "t/dom_extent.ent">
 %globalInfo;
 
]>
<book>
</book>
EOF

#
# Tell XML::Parser to parse the external entities (ParseParamEnt => 1)
# Tell XML::DOM::Parser to 'hide' the contents of the external entities
# so you see '%globalInfo;' when printing.
my $parser = new XML::DOM::Parser( 
				   ParseParamEnt => 1,
				   ExpandParamEnt => 0,
				   ErrorContext => 5);

my $dom = $parser->parse ($xml);
my $domstr = $dom->toString;

# Compare output with original file
assert_ok ($domstr eq $xml);

print "$domstr\n$xml\n";
