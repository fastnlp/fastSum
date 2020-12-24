BEGIN {print "1..3\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
use CheckAncestors;
use CmpDOM;
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
<?xml version="1.0"?>
<elt attr="foo &amp; bar ]]>"/>
END

my $expected = <<END;
<?xml version="1.0"?>
<elt attr="foo &amp; bar ]]&gt;"/>
END

my $parser = new XML::DOM::Parser;
my $doc = $parser->parse ($str);
assert_ok (not $@);

my $out = $doc->toString;
assert_ok ($out eq $expected);
