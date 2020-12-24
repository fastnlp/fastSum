BEGIN {print "1..2\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
$loaded = 1;
print "ok 1\n";

my $test = 1;
sub assert_ok
{
    my $ok = shift;
    print "not " unless $ok;
    ++$test;
    print "ok $test\n";
    $ok;
}

#Test 2

my $str = <<END;
<southpark>
<chef voice-type="bass">
Hello children
</chef>
<children>
Hello Chef
<kenny voice-type="muffled">
Whoowhoo whoo
</kenny>
<cartman ass="fat">
Shut up you loser
</cartman>
<kyle>
Cartman, you fat ass
</kyle>
</children>
</southpark>
END

# This example has attribute names with "-" (non alphanumerics)
# A previous bug caused attributes with non-alphanumeric names to always
# be interpreted as default attribute values. When printing out the document
# they would not be printed, because default attributes aren't printed.
my $parser = new XML::DOM::Parser;
my $doc = $parser->parse ($str);

my $out = $doc->toString;
$out =~ tr/\012/\n/;
assert_ok ($out eq $str);
