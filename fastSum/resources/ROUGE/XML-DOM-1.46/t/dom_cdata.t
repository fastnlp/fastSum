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
<book>
<award>
    <![CDATA[Trenton Literary Review Honorable Mention]]>
</award>
</book>
END

my $oldStr = <<END;
<?xml version="1.0"?>
<book>
<award>
    Trenton Literary Review Honorable Mention
</award>
</book>
END

# Keep CDATASections intact. Without this option set (default), it will convert
# CDATASections to Text nodes. The KeepCDATA option is only supported
# with XML::Parser versions 2.19 and up.
my $parser = new XML::DOM::Parser (KeepCDATA => 1);
my $doc = $parser->parse ($str);
assert_ok (not $@);

my $out = $doc->toString;
$out =~ tr/\012/\n/;
my $result = ($XML::Parser::VERSION >= 2.19) ? $str : $oldStr;
assert_ok ($out eq $result);
