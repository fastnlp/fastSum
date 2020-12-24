BEGIN {print "1..16\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
use CheckAncestors;
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
<chef>
Hello children
</chef>
<children>
Hello Chef
<kenny>
Whoowhoo whoo
</kenny>
<cartman>
Shut up you loser
</cartman>
<kyle>
Cartman, you fat ass
</kyle>
</children>
</southpark>
END

my $parser = new XML::DOM::Parser;
my $doc = $parser->parse ($str);

my $chef = $doc->getElementsByTagName ("chef")->item(0);
my $kenny = $doc->getElementsByTagName ("kenny")->item(0);
my $children = $doc->getElementsByTagName ("children")->item(0);

my $stan = $doc->createElement ("stan");
$children->appendChild ($stan);
my $snap1 =$doc->toString;

my $stanlist = $doc->getElementsByTagName ("stan");
assert_ok ($stanlist->getLength == 1);

$children->appendChild ($stan);
$stanlist = $doc->getElementsByTagName ("stan");
assert_ok ($stanlist->getLength == 1);

my $snap2 = $doc->toString;
assert_ok ($snap1 eq $snap2);

# can't add Attr node directly to Element
my $attr = $doc->createAttribute ("hey", "you");
eval {
    $kenny->appendChild ($attr);
};
assert_ok ($@);

$kenny->appendChild ($stan);
assert_ok ($kenny == $stan->getParentNode);

# force hierarchy exception
eval {
    $stan->appendChild ($kenny);
};
assert_ok ($@);

# force hierarchy exception
eval {
    $stan->appendChild ($stan);
};
assert_ok ($@);

my $frag = $doc->createDocumentFragment;
$frag->appendChild ($stan);
$frag->appendChild ($kenny);
$chef->appendChild ($frag);
assert_ok ($frag->getElementsByTagName ("*")->getLength == 0);
assert_ok (not defined $frag->getParentNode);

my $kenny2 = $chef->removeChild ($kenny);
assert_ok ($kenny == $kenny2);
assert_ok (!defined $kenny->getParentNode);

# force exception - can't have 2 element nodes in a document
eval {
    $doc->appendChild ($kenny);
};
assert_ok ($@);

$doc->getDocumentElement->appendChild ($kenny);
$kenny2 = $doc->getDocumentElement->replaceChild ($stan, $kenny);
assert_ok ($kenny == $kenny2);

$doc->getDocumentElement->appendChild ($kenny);

assert_ok (CheckAncestors::doit ($doc));

$str = $doc->toString;
$str =~ tr/\012/\n/;

my $end = <<END;
<southpark>
<chef>
Hello children
</chef>
<children>
Hello Chef

<cartman>
Shut up you loser
</cartman>
<kyle>
Cartman, you fat ass
</kyle>
</children>
<stan/><kenny>
Whoowhoo whoo
</kenny></southpark>
END

assert_ok ($str eq $end);
